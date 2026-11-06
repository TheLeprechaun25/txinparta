/*
 * pultsito_shim.c -- a flat C entry point so Python (ctypes) can run Pultsito.
 *
 * Pultsito's own executable reads TOML/text files and returns only timings
 * (simulate_batches() resets the per-batch spike counts and never hands them
 * back). This shim builds Pultsito's structures straight from arrays and runs
 * the batch CPU engine step by step with Pultsito's own functions:
 *
 *     load_batch_time_step_in_SNN_batch  -> inputs for step t
 *     compute_input_current_batch        -> synaptic currents
 *     compute_LIF_V_batch                -> membrane update
 *     process_neuron_firing_batch        -> spikes, reset, refractory
 *
 * After each step it reads the spike matrix to accumulate spike counts and
 * (optionally) time-binned counts per sample.
 *
 * Workaround for an upstream issue (keep until fixed in Pultsito):
 *  cpy_snn() allocates the batched arrays without freeing the originals,
 *  so the shim frees them.
 *
 * The neuron model is Pultsito's: alpha = 0.95 (tau = 20), and the
 * resistance is ignored.
 */
#include <stdlib.h>
#include <string.h>
#include "snn_library.h"

/* Defined in Pultsito's simulations.c but not exported by any header. */
void compute_input_current_batch(GPU_SNN_t *snn, simulation_configuration_t *conf, size_t t, size_t gt);
void compute_LIF_V_batch(GPU_SNN_t *snn, simulation_configuration_t *conf, size_t t, size_t gt);
void process_neuron_firing_batch(GPU_SNN_t *snn, simulation_configuration_t *conf, GPU_results_t *results, size_t t, size_t gt);
void reinitialize_LIF_neurons_batch(GPU_SNN_t *snn, simulation_configuration_t *conf);
void reinitialize_synapses_batch(GPU_SNN_t *snn, simulation_configuration_t *conf);
void reinitialize_spk_matrix_batch(GPU_SNN_t *snn, simulation_configuration_t *conf);
tmp_batch_cpu_t *initialize_batch_matrix(GPU_SNN_t *snn, GPU_dataset_t *dataset, simulation_configuration_t *conf, size_t bidx);
void load_batch_time_step_in_SNN_batch(GPU_SNN_t *snn, GPU_dataset_t *dataset, simulation_configuration_t *conf, tmp_batch_cpu_t *batch_data, size_t bidx, size_t t);
void deallocate_network_cpu(GPU_SNN_t *snn);

/* Build Pultsito's construction lists from flat, post-sorted synapse arrays. */
static network_construction_lists_t *make_lists(
    size_t n_inputs, size_t n_neurons, size_t n_synapses,
    const float *v_thresh, const float *v_rest, const int *refractory,
    const int *pre, const int *post, const float *w, const int *delay)
{
    network_construction_lists_t *L = calloc(1, sizeof(*L));
    L->n_neurons = n_neurons;
    L->n_input_neurons = n_inputs;
    L->n_synapses = n_synapses;
    L->v_list = malloc(n_neurons * sizeof(double));
    L->v_thres_list = malloc(n_neurons * sizeof(double));
    L->v_rest_list = malloc(n_neurons * sizeof(double));
    L->R_list = malloc(n_neurons * sizeof(double));
    L->r_time_list = malloc(n_neurons * sizeof(int));
    for (size_t i = 0; i < n_neurons; i++) {
        L->v_list[i] = v_rest[i];
        L->v_thres_list[i] = v_thresh[i];
        L->v_rest_list[i] = v_rest[i];
        L->R_list[i] = 1.0;
        L->r_time_list[i] = refractory[i];
    }
    L->weight_list = malloc(n_synapses * sizeof(double));
    L->delay_list = malloc(n_synapses * sizeof(int));
    size_t max_delay = 0;
    for (size_t s = 0; s < n_synapses; s++) {
        L->weight_list[s] = w[s];
        L->delay_list[s] = delay[s];
        if ((size_t)delay[s] > max_delay) max_delay = delay[s];
        if ((size_t)pre[s] < n_inputs) L->n_input_synapses++;
    }
    L->max_delay = max_delay + 1; /* ring-buffer length, as in load_data.c */

    /* synaptic_connections[i] = [n_entries, pre_0, count_0, pre_1, count_1, ...];
       one entry per synapse keeps Pultsito's synapse order equal to ours. */
    size_t *n_in = calloc(n_neurons, sizeof(size_t));
    for (size_t s = 0; s < n_synapses; s++) n_in[post[s]]++;
    L->synaptic_connections = malloc(n_neurons * sizeof(int *));
    size_t s = 0;
    for (size_t i = 0; i < n_neurons; i++) {
        int *row = malloc((1 + 2 * n_in[i]) * sizeof(int));
        row[0] = (int)n_in[i];
        for (size_t j = 0; j < n_in[i]; j++, s++) {
            row[1 + 2 * j] = pre[s];
            row[2 + 2 * j] = 1;
        }
        L->synaptic_connections[i] = row;
    }
    free(n_in);
    return L;
}

static void free_lists(network_construction_lists_t *L) {
    for (size_t i = 0; i < L->n_neurons; i++) free(L->synaptic_connections[i]);
    free(L->synaptic_connections);
    free(L->v_list); free(L->v_thres_list); free(L->v_rest_list); free(L->R_list);
    free(L->r_time_list); free(L->weight_list); free(L->delay_list);
    free(L);
}

/* Replicate the network B times (one copy per batch element), freeing the originals. */
static void batch_copies(GPU_SNN_t *snn, simulation_configuration_t *conf) {
    void *old[] = {snn->v, snn->arrI, snn->r_period_remain, snn->post_fired, snn->post_trace,
                   snn->w, snn->dw, snn->pre_trace, snn->pre_fired, snn->spk_matrix};
    cpy_snn(snn, conf);
    for (size_t k = 0; k < sizeof(old) / sizeof(old[0]); k++) free(old[k]);
}

/*
 * Simulate n_samples samples.
 *   inputs: n_spikes[n_samples * n_inputs] spikes per (sample, channel), and
 *           spike_times = all spike times, ordered by sample, then channel.
 *   outputs: counts[n_samples * n_neurons]
 *            binned[n_samples * n_bins * n_neurons] (may be NULL; n_bins = ceil(T / bin_size))
 * Returns 0 on success.
 */
int darwinsnn_pultsito_run(
    size_t n_inputs, size_t n_neurons, size_t n_synapses,
    const float *v_thresh, const float *v_rest, const int *refractory,
    const int *pre, const int *post, const float *w, const int *delay,
    size_t n_samples, const size_t *n_spikes, const size_t *spike_times,
    size_t time_steps, size_t batch_size, int n_threads, size_t bin_size,
    int *counts, int *binned)
{
    simulation_configuration_t conf;
    memset(&conf, 0, sizeof(conf));
    conf.batch_size = batch_size;
    conf.n_process = n_threads > 0 ? (size_t)n_threads : 1;
    conf.time_steps = time_steps;
    conf.max_input_spikes = time_steps;  /* length of the per-batch input matrix */
    conf.learn = 0;

    network_construction_lists_t *L = make_lists(n_inputs, n_neurons, n_synapses, v_thresh,
                                                 v_rest, refractory, pre, post, w, delay);
    GPU_SNN_t *snn = initialize_network_cpu(&conf, L);
    batch_copies(snn, &conf);

    /* dataset in Pultsito's flat layout */
    GPU_dataset_t ds;
    memset(&ds, 0, sizeof(ds));
    ds.n_samples = n_samples;
    ds.n_features = n_inputs;
    ds.n_spikes_per_feature = (size_t *)n_spikes;
    ds.feature_offset = malloc(n_samples * n_inputs * sizeof(size_t));
    size_t off = 0;
    for (size_t k = 0; k < n_samples * n_inputs; k++) {
        ds.feature_offset[k] = off;
        off += n_spikes[k];
    }
    ds.n_spikes = off;
    ds.spikes = (size_t *)spike_times;

    GPU_results_t *res = initialize_batch_results_cpu(&conf, n_neurons, batch_size, 1);
    const size_t N = n_neurons, iN = n_inputs, B = batch_size, LT = snn->LT;
    const size_t nb = bin_size ? (time_steps + bin_size - 1) / bin_size : 0;
    const size_t n_batches = (n_samples + B - 1) / B;

    for (size_t bidx = 0; bidx < n_batches; bidx++) {
        reinitialize_LIF_neurons_batch(snn, &conf);
        reinitialize_synapses_batch(snn, &conf);
        reinitialize_spk_matrix_batch(snn, &conf);
        tmp_batch_cpu_t *batch_in = initialize_batch_matrix(snn, &ds, &conf, bidx);

        for (size_t t = 0; t < time_steps; t++) {
            size_t lt = t % LT;
            char *row = snn->spk_matrix + (iN + N) * B * lt;
            load_batch_time_step_in_SNN_batch(snn, &ds, &conf, batch_in, bidx, t);
            compute_input_current_batch(snn, &conf, lt, t);
            compute_LIF_V_batch(snn, &conf, lt, t);
            process_neuron_firing_batch(snn, &conf, res, lt, t);

            for (size_t b = 0; b < B; b++) {
                size_t sidx = bidx * B + b;
                if (sidx >= n_samples) break;
                for (size_t i = 0; i < N; i++) {
                    if (row[B * (iN + i) + b]) {
                        counts[sidx * N + i]++;
                        if (binned) binned[(sidx * nb + t / bin_size) * N + i]++;
                    }
                }
            }
        }
        free(batch_in->spikes);
        free(batch_in);
    }

    free(res->n_spks);
    free(res);
    free(ds.feature_offset);
    deallocate_network_cpu(snn);
    free_lists(L);
    return 0;
}
