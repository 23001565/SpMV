// csb_morton_trans_recursive.cpp
// Build: g++ -O3 -march=native -fopenmp csb_morton_trans_recursive.cpp -o csb_morton_trans_recursive

#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

// COO tuple
struct COO { int nrows=0, ncols=0; vector<int> r,c; vector<double> v; };

// read simple COO file: first line "n m nnz" then nnz lines "r c val" (0-based)
COO read_coo_header(const string &fname) {
    ifstream fin(fname);
    if (!fin) { cerr<<"Cannot open "<<fname<<"\n"; exit(1); }
    int n,m,nnz;
    fin >> n >> m >> nnz;
    COO C; C.nrows = n; C.ncols = m;
    C.r.resize(nnz); C.c.resize(nnz); C.v.resize(nnz);
    for (int k=0;k<nnz;k++) {
        if (!(fin >> C.r[k] >> C.c[k] >> C.v[k])) {
            cerr<<"Malformed COO at index "<<k<<"\n"; exit(1);
        }
    }
    return C;
}

// interleave bits (Morton) for up to 16 bits (Beta <= 65536)
static inline size_t morton_key_u16(unsigned r, unsigned c) {
    size_t key = 0;
    for (int b=0;b<16;b++) {
        key |= ((size_t)((r>>b)&1) << (2*b + 1));
        key |= ((size_t)((c>>b)&1) << (2*b));
    }
    return key;
}

int genBeta(int n) {
    if (n <= 1) return 1;
    int sq = (int)floor(sqrt((double)n));
    int b = 1;
    while (b <= sq) b <<= 1;
    return max(1, b>>1);
}

// CSB packed representation
struct CSB {
    int N=0, M=0;
    int Beta=0;
    int NB_r=0, NB_c=0;
    vector<size_t> BlkPtr; // size NB + 1
    vector<int> RowIdx;    // local row inside block
    vector<int> ColIdx;    // local col inside block
    vector<double> Val;    // values, all blocks concatenated
};

// Build CSB: distribute to block buckets, sort each block by Morton key, then flatten
CSB build_csb_packed(const COO &C, int userBeta=0) {
    CSB B;
    B.N = C.nrows; B.M = C.ncols;
    int maxNM = max(B.N, B.M);
    B.Beta = (userBeta>0)?userBeta:genBeta(maxNM);
    if (B.Beta < 1) B.Beta = 1;
    B.NB_r = (B.N + B.Beta - 1) / B.Beta;
    B.NB_c = (B.M + B.Beta - 1) / B.Beta;
    size_t NumBlocks = (size_t)B.NB_r * (size_t)B.NB_c;

    // temporary per-block buckets of indices
    vector<vector<int>> bucket_idx(NumBlocks);
    bucket_idx.shrink_to_fit();
    int nnz = (int)C.v.size();
    for (int k=0;k<nnz;k++) {
        int bi = C.r[k] / B.Beta;
        int bj = C.c[k] / B.Beta;
        size_t b = (size_t)bi * B.NB_c + bj;
        bucket_idx[b].push_back(k);
    }

    // compute BlkPtr
    B.BlkPtr.assign(NumBlocks+1, 0);
    for (size_t b=0;b<NumBlocks;b++) B.BlkPtr[b+1] = B.BlkPtr[b] + bucket_idx[b].size();
    size_t totalNNZ = B.BlkPtr[NumBlocks];

    B.Val.resize(totalNNZ);
    B.RowIdx.resize(totalNNZ);
    B.ColIdx.resize(totalNNZ);

    // flatten blocks with Morton sorting inside each block
    size_t off = 0;
    for (size_t b=0;b<NumBlocks;b++) {
        size_t bn = bucket_idx[b].size();
        if (bn==0) continue;
        vector<size_t> order(bn);
        for (size_t t=0;t<bn;++t) order[t]=t;
        int bi = (int)(b / B.NB_c);
        int bj = (int)(b % B.NB_c);
        int row_off = bi * B.Beta;
        int col_off = bj * B.Beta;
        sort(order.begin(), order.end(), [&](size_t a, size_t b2){
            int idxA = bucket_idx[b][a];
            int idxB = bucket_idx[b][b2];
            unsigned rA = (unsigned)(C.r[idxA] - row_off);
            unsigned cA = (unsigned)(C.c[idxA] - col_off);
            unsigned rB = (unsigned)(C.r[idxB] - row_off);
            unsigned cB = (unsigned)(C.c[idxB] - col_off);
            return morton_key_u16(rA,cA) < morton_key_u16(rB,cB);
        });
        for (size_t t=0;t<bn;++t) {
            int idx = bucket_idx[b][order[t]];
            int local_r = C.r[idx] - row_off;
            int local_c = C.c[idx] - col_off;
            B.RowIdx[off] = local_r;
            B.ColIdx[off] = local_c;
            B.Val[off] = C.v[idx];
            ++off;
        }
    }
    return B;
}

// --- blockV: recursive on [s..e] within a block, Dim = block dimension
void blockV_recursive(const CSB &B,
                      size_t s, size_t e, int Dim,
                      const double *Xblock, double *Yblock,
                      int task_cutoff_dim = 152)
{
    if (s > e) return;
    size_t len = e - s + 1;
    if (len <= (size_t)Dim || Dim <= task_cutoff_dim) {
        for (size_t k = s; k <= e; ++k) {
            Yblock[ B.RowIdx[k] ] += B.Val[k] * Xblock[ B.ColIdx[k] ];
        }
        return;
    }

    int Half = Dim >> 1;
    size_t s2 = e + 1;
    size_t lo = s, hi = e;
    while (lo <= hi) {
        size_t mid = (lo+hi)>>1;
        if (B.RowIdx[mid] & Half) { s2 = mid; if (mid==0) break; hi = mid-1; }
        else lo = mid+1;
    }
    size_t s1 = s2;
    if (s2 > s) {
        lo = s; hi = s2 - 1;
        while (lo <= hi) {
            size_t mid = (lo+hi)>>1;
            if (B.ColIdx[mid] & Half) { s1 = mid; if (mid==0) break; hi = mid-1; }
            else lo = mid+1;
        }
    }
    size_t s3 = e + 1;
    if (s2 <= e) {
        lo = s2; hi = e;
        while (lo <= hi) {
            size_t mid = (lo+hi)>>1;
            if (B.ColIdx[mid] & Half) { s3 = mid; if (mid==0) break; hi = mid-1; }
            else lo = mid+1;
        }
    }

    #pragma omp taskgroup
    {
        #pragma omp task shared(B)
        blockV_recursive(B, s, (s1==0?0:s1-1), Half, Xblock, Yblock, task_cutoff_dim);

        #pragma omp task shared(B)
        if (s3 <= e) blockV_recursive(B, s3, e, Half, Xblock, Yblock, task_cutoff_dim);

        #pragma omp task shared(B)
        if (s1 <= (s2==0?0:s2-1)) blockV_recursive(B, s1, (s2==0?0:s2-1), Half, Xblock, Yblock, task_cutoff_dim);

        #pragma omp task shared(B)
        if (s2 <= (s3==0?0:s3-1)) blockV_recursive(B, s2, (s3==0?0:s3-1), Half, Xblock, Yblock, task_cutoff_dim);
    }
}

// --- blockV_recursive_trans: transpose version (swap roles of row/col inside block)
void blockV_recursive_trans(const CSB &B,
                      size_t s, size_t e, int Dim,
                      const double *Xblock, double *Yblock,
                      int task_cutoff_dim = 152)
{
    if (s > e) return;
    size_t len = e - s + 1;
    if (len <= (size_t)Dim || Dim <= task_cutoff_dim) {
        // direct: swapped access
        for (size_t k = s; k <= e; ++k) {
            int lr = B.RowIdx[k];
            int lc = B.ColIdx[k];
            Yblock[ lc ] += B.Val[k] * Xblock[ lr ];
        }
        return;
    }

    int Half = Dim >> 1;
    // s2: first index where RowIdx bit Half is 1
    size_t s2 = e + 1;
    size_t lo = s, hi = e;
    while (lo <= hi) {
        size_t mid = (lo+hi)>>1;
        if (B.RowIdx[mid] & Half) { s2 = mid; hi = mid-1; }
        else lo = mid+1;
    }
    // s1: first index in [s, s2-1] where ColIdx bit Half is 1
    size_t s1 = s2;
    if (s2 > s) {
        lo = s; hi = s2 - 1;
        while (lo <= hi) {
            size_t mid = (lo+hi)>>1;
            if (B.ColIdx[mid] & Half) { s1 = mid; hi = mid-1; }
            else lo = mid+1;
        }
    }
    // s3: first index in [s2, e] where ColIdx bit Half is 1
    size_t s3 = e + 1;
    if (s2 <= e) {
        lo = s2; hi = e;
        while (lo <= hi) {
            size_t mid = (lo+hi)>>1;
            if (B.ColIdx[mid] & Half) { s3 = mid; hi = mid-1; }
            else lo = mid+1;
        }
    }

    #pragma omp taskgroup
    {
        #pragma omp task shared(B)
        blockV_recursive_trans(B, s, (s1==0?0:s1-1), Half, Xblock, Yblock, task_cutoff_dim);

        #pragma omp task shared(B)
        if (s1 <= (s2==0?0:s2-1))
            blockV_recursive_trans(B, s1, (s2==0?0:s2-1), Half, Xblock, Yblock, task_cutoff_dim);

        #pragma omp task shared(B)
        if (s2 <= (s3==0?0:s3-1))
            blockV_recursive_trans(B, s2, (s3==0?0:s3-1), Half, Xblock, Yblock, task_cutoff_dim);

        #pragma omp task shared(B)
        if (s3 <= e)
            blockV_recursive_trans(B, s3, e, Half, Xblock, Yblock, task_cutoff_dim);
    }
}

// blockRowV: process splits R vector for block-row bi (forward)
void blockRowV_recursive(const CSB &B, int bi,
                         const vector<int> &R,
                         const double *Xfull, int Xlen,
                         double *Yblock, int Ylen,
                         int task_cutoff_dim = 152)
{
    int Rlen = (int)R.size();
    if (Rlen == 2) {
        int l = R[0] + 1;
        int r = R[1];
        if (l > r) return;
        for (int bj = l; bj <= r; ++bj) {
            size_t b = (size_t)bi * B.NB_c + bj;
            size_t s = B.BlkPtr[b];
            size_t e = B.BlkPtr[b+1];
            if (s >= e) continue;
            const double *Xblk = Xfull + (size_t)bj * B.Beta;
            blockV_recursive(B, s, e-1, B.Beta, Xblk, Yblock, task_cutoff_dim);
        }
        return;
    }
    int Mid = (Rlen & 1) ? (Rlen >> 1) : ((Rlen >> 1) - 1);
    int XMid = B.Beta * (R[Mid] - R[0]);
    vector<double> Z(Ylen, 0.0);

    #pragma omp taskgroup
    {
        #pragma omp task
        blockRowV_recursive(B, bi, vector<int>(R.begin(), R.begin()+Mid+1),
                            Xfull, XMid, Yblock, Ylen, task_cutoff_dim);
        #pragma omp task
        blockRowV_recursive(B, bi, vector<int>(R.begin()+Mid, R.end()),
                            Xfull + XMid, Xlen - XMid, Z.data(), Ylen, task_cutoff_dim);
    }
    for (int i=0;i<Ylen;i++) Yblock[i] += Z[i];
}

// blockRowV_recursive_trans: transpose-side equivalent (operates per block-column bcj)
void blockRowV_recursive_trans(const CSB &B, int bcj,
                               const vector<int> &R,
                               const double *Xfull, int Xlen,
                               double *Yblock, int Ylen,
                               int task_cutoff_dim = 152)
{
    int Rlen = (int)R.size();
    if (Rlen == 2) {
        int l = R[0] + 1;
        int r = R[1];
        if (l > r) return;
        // iterate over block-rows (bri) in [l..r]
        for (int bri = l; bri <= r; ++bri) {
            size_t b = (size_t)bri * B.NB_c + bcj;
            size_t s = B.BlkPtr[b];
            size_t e = B.BlkPtr[b+1];
            if (s >= e) continue;
            const double *Xblk = Xfull + (size_t)bri * B.Beta; // X over rows
            blockV_recursive_trans(B, s, e-1, B.Beta, Xblk, Yblock, task_cutoff_dim);
        }
        return;
    }

    int Mid = (Rlen & 1) ? (Rlen >> 1) : ((Rlen >> 1) - 1);
    int XMid = B.Beta * (R[Mid] - R[0]); // same semantics for splitting
    vector<double> Z(Ylen, 0.0);

    #pragma omp taskgroup
    {
        #pragma omp task
        blockRowV_recursive_trans(B, bcj, vector<int>(R.begin(), R.begin()+Mid+1),
                                  Xfull, XMid, Yblock, Ylen, task_cutoff_dim);
        #pragma omp task
        blockRowV_recursive_trans(B, bcj, vector<int>(R.begin()+Mid, R.end()),
                                  Xfull + XMid, Xlen - XMid, Z.data(), Ylen, task_cutoff_dim);
    }
    for (int i=0;i<Ylen;i++) Yblock[i] += Z[i];
}

// High-level SpMV A*x using CSB: parallel over block-rows via OpenMP single+tasks
void csb_spmv_ax(const CSB &B, const vector<double> &x, vector<double> &y) {
    y.assign(B.N, 0.0);
    int NB_r = B.NB_r;
    int NB_c = B.NB_c;
    int Beta = B.Beta;

    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            for (int bri=0; bri<NB_r; ++bri) {
                #pragma omp task firstprivate(bri)
                {
                    vector<int> R; R.reserve(NB_c+2);
                    R.push_back(-1);
                    size_t cnt = 0;
                    for (int bj=0; bj<NB_c-1; ++bj) {
                        size_t bidx = (size_t)bri * NB_c + bj;
                        cnt += (B.BlkPtr[bidx+1] - B.BlkPtr[bidx]);
                        size_t nextcnt = (B.BlkPtr[bidx+2] - B.BlkPtr[bidx+1]);
                        if (cnt + nextcnt > (size_t)Beta) { R.push_back(bj); cnt = 0; }
                    }
                    R.push_back(NB_c - 1);

                    int Ylen = min(Beta, B.N - bri*Beta);
                    vector<double> Yblock(Ylen, 0.0);

                    blockRowV_recursive(B, bri, R, x.data(), (int)x.size(), Yblock.data(), Ylen);

                    int y_offset = bri * Beta;
                    for (int i=0;i<Ylen;i++) {
                        y[y_offset + i] += Yblock[i];
                    }
                }
            }
        }
    }
}

// CSB transpose A^T * x: recursive per block-column using blockRowV_recursive_trans
void csb_spmv_atx(const CSB &B, const vector<double> &x, vector<double> &y) {
    y.assign(B.M, 0.0);
    int NB_r = B.NB_r;
    int NB_c = B.NB_c;
    int Beta = B.Beta;

    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            for (int bcj=0; bcj<NB_c; ++bcj) {
                #pragma omp task firstprivate(bcj)
                {
                    // build R splits over block-rows for this block-column bcj
                    vector<int> R; R.reserve(NB_r+2);
                    R.push_back(-1);
                    size_t cnt = 0;
                    for (int bri=0; bri<NB_r-1; ++bri) {
                        size_t bidx = (size_t)bri * NB_c + bcj;
                        cnt += (B.BlkPtr[bidx+1] - B.BlkPtr[bidx]);
                        size_t nextcnt = (B.BlkPtr[bidx+NB_c+1] - B.BlkPtr[bidx+NB_c]); // careful index
                        // The above expression computes next block-row's block pointer difference:
                        // block at (bri+1, bcj) has index (bidx + NB_c)
                        if (cnt + nextcnt > (size_t)Beta) { R.push_back(bri); cnt = 0; }
                    }
                    R.push_back(NB_r - 1);

                    int Ylen = min(Beta, B.M - bcj*Beta);
                    vector<double> Yblock(Ylen, 0.0);

                    // Xfull for transpose is x over rows (length B.N)
                    blockRowV_recursive_trans(B, bcj, R, x.data(), (int)x.size(), Yblock.data(), Ylen);

                    int out_off = bcj * Beta;
                    for (int i=0;i<Ylen;i++) y[out_off + i] += Yblock[i];
                } // end task
            } // end for bcj
        } // end single
    } // end parallel
}



// ------------------ MAIN ------------------
int main(int argc, char** argv) {

    if (argc < 2) {
        cerr<<"Usage: ./csb_morton matrix.coo [num_threads] [beta]\n";
        return 1;
    }

    string fname = argv[1];
    int num_threads = (argc>=3 ? atoi(argv[2]) : 0);
    int userBeta   = (argc>=4 ? atoi(argv[3]) : 0);

    if (num_threads>0) omp_set_num_threads(num_threads);

    cout<<"Threads: "<<(num_threads>0?num_threads:omp_get_max_threads())<<"\n";

    // Read COO
    double t0=omp_get_wtime();
    COO C=read_coo_header(fname);
    double t1=omp_get_wtime();
    cout<<"Read COO: N="<<C.nrows<<" M="<<C.ncols<<" nnz="<<C.v.size()
        <<"  time="<<(t1-t0)<<" s\n";

    // Build CSB
    double t2=omp_get_wtime();
    CSB B=build_csb_packed(C,userBeta);
    double t3=omp_get_wtime();
    cout<<"Built CSB: Beta="<<B.Beta<<" NB_r="<<B.NB_r<<" NB_c="<<B.NB_c
        <<" nnz_total="<<B.Val.size()<<"  build_time="<<(t3-t2)<<" s\n";

    // Fill x with 1.0
    vector<double> x_ax(B.M, 1.0);
    vector<double> x_atx(B.N, 1.0);
    vector<double> y, yt;

    // Run once: AX
    double s=omp_get_wtime();
    csb_spmv_ax(B, x_ax, y);
    double t_ax=omp_get_wtime()-s;

    // Run once: ATX
    s=omp_get_wtime();
    csb_spmv_atx(B, x_atx, yt);
    double t_atx=omp_get_wtime()-s;

    cout<<fixed<<setprecision(6)
        <<"\nCSB A*x   time="<<t_ax<<" s \n"
        <<"CSB A^T*x time="<<t_atx<<" s \n";

    return 0;
}