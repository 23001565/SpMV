// file: spmv_openmp.cpp
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <omp.h>
using namespace std;

// ====================== COO Structure ======================
struct COOTuple {
    int row, col;
    double val;
};

// =================== Read COO File (.coo) ===================
void readCOO(const string& filename, int& num_rows, int& num_cols, int& nnz,
             vector<COOTuple>& entries)
{
    ifstream fin(filename);
    if (!fin) {
        cerr << "Error: Cannot open file " << filename << "\n";
        exit(1);
    }

    fin >> num_rows >> num_cols >> nnz;
    entries.resize(nnz);

    for (int i = 0; i < nnz; i++)
        fin >> entries[i].row >> entries[i].col >> entries[i].val;
}

// =================== COO → CSR Conversion ===================
void COO_to_CSR(int num_rows, int nnz,
                const vector<COOTuple>& entries,
                vector<int>& row_ptr,
                vector<int>& col_ind,
                vector<double>& values)
{
    row_ptr.assign(num_rows + 1, 0);
    col_ind.resize(nnz);
    values.resize(nnz);

    // Count nonzeros per row
    for (const auto &e : entries)
        row_ptr[e.row]++;

    // Prefix sum
    for (int i = 1; i <= num_rows; i++)
        row_ptr[i] += row_ptr[i - 1];

    // Temporary for placement
    vector<int> current = row_ptr;

    for (const auto &e : entries) {
        int pos = --current[e.row];
        col_ind[pos] = e.col;
        values[pos] = e.val;
    }
}

// ===================== SpMV: y = A*x (OpenMP parallel) ========================
void SpMV_parallel(int num_rows,
          const vector<int>& row_ptr,
          const vector<int>& col_ind,
          const vector<double>& values,
          const vector<double>& x,
          vector<double>& y)
{
    y.assign(num_rows, 0.0);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_rows; i++) {
        double sum = 0.0;
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++)
            sum += values[j] * x[col_ind[j]];
        y[i] = sum;
    }
}

// ================= SpMV_T: y = Aᵗ * x (parallel, private buffers) =============
void SpMV_T_parallel(int num_rows, int num_cols,
            const vector<int>& row_ptr,
            const vector<int>& col_ind,
            const vector<double>& values,
            const vector<double>& x,
            vector<double>& y)
{
    y.assign(num_cols, 0.0);

    int nthreads = omp_get_max_threads();
    vector<vector<double>> priv(nthreads, vector<double>(num_cols, 0.0));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        vector<double> &yloc = priv[tid];

        #pragma omp for schedule(static)
        for (int row = 0; row < num_rows; row++) {
            double xv = x[row];
            for (int j = row_ptr[row]; j < row_ptr[row + 1]; j++)
                yloc[col_ind[j]] += values[j] * xv;
        }
    }

    // Combine thread copies
    for (int t = 0; t < nthreads; t++)
        for (int c = 0; c < num_cols; c++)
            y[c] += priv[t][c];
}

// =========== SpMV_T (atomic version): y = Aᵗ * x using atomics =================
void SpMV_T_parallel_atomic(int num_rows, int num_cols,
            const vector<int>& row_ptr,
            const vector<int>& col_ind,
            const vector<double>& values,
            const vector<double>& x,
            vector<double>& y)
{
    y.assign(num_cols, 0.0);

    #pragma omp parallel for schedule(static)
    for (int row = 0; row < num_rows; row++) {
        double xv = x[row];
        for (int j = row_ptr[row]; j < row_ptr[row + 1]; j++) {
            int col = col_ind[j];
            double val = values[j] * xv;

            #pragma omp atomic
            y[col] += val;
        }
    }
}



// ============================= MAIN ================================
int main(int argc, char** argv)
{
    string filename = "matrix.coo";
    int num_threads = 1;   // default

    if (argc >= 2)
        filename = argv[1];

    if (argc >= 3)
        num_threads = atoi(argv[2]);   // user-specified thread count

    if (num_threads <= 0)
        num_threads = 1;

    omp_set_num_threads(num_threads);

    int num_rows = 0, num_cols = 0, nnz = 0;
    vector<COOTuple> entries;

    readCOO(filename, num_rows, num_cols, nnz, entries);

    vector<int> row_ptr, col_ind;
    vector<double> values;
    COO_to_CSR(num_rows, nnz, entries, row_ptr, col_ind, values);

    cout << "Matrix: " << num_rows << " x " << num_cols
         << ", nnz = " << nnz << "\n";
    cout << "Requested threads = " << num_threads << "\n";
    cout << "OpenMP using      = " << omp_get_max_threads() << "\n";

    vector<double> x(max(num_rows, num_cols), 1.0);
    vector<double> y1, y2, y3;

    // Warm-up
    SpMV_parallel(num_rows, row_ptr, col_ind, values, x, y1);
    SpMV_T_parallel(num_rows, num_cols, row_ptr, col_ind, values, x, y2);
    SpMV_T_parallel_atomic(num_rows, num_cols, row_ptr, col_ind, values, x, y3);

    // Benchmark 1: SpMV
    auto t1 = chrono::high_resolution_clock::now();
    SpMV_parallel(num_rows, row_ptr, col_ind, values, x, y1);
    auto t2 = chrono::high_resolution_clock::now();

    // Benchmark 2: SpMV_T (private buffers)
    auto t3 = chrono::high_resolution_clock::now();
    SpMV_T_parallel(num_rows, num_cols, row_ptr, col_ind, values, x, y2);
    auto t4 = chrono::high_resolution_clock::now();

    // Benchmark 3: SpMV_T (atomic)
    auto t5 = chrono::high_resolution_clock::now();
    SpMV_T_parallel_atomic(num_rows, num_cols, row_ptr, col_ind, values, x, y3);
    auto t6 = chrono::high_resolution_clock::now();

    double time_spmv       = chrono::duration<double, milli>(t2 - t1).count();
    double time_spmvt_priv = chrono::duration<double, milli>(t4 - t3).count();
    double time_spmvt_atm  = chrono::duration<double, milli>(t6 - t5).count();

    cout << "\n===== Timing Results =====\n";
    cout << "SpMV (parallel)               : " << time_spmv       << " ms  \n";
    cout << "SpMV_T (private-buffer)       : " << time_spmvt_priv << " ms  \n";
    cout << "SpMV_T (atomic)               : " << time_spmvt_atm  << " ms  \n";

    return 0;
}
