#include "Fwbw.h"
#include "Debug.h"
#include "DBReader.h"
#include "DBWriter.h"
#include "IndexReader.h"
#include "SubstitutionMatrix.h"
#include "Matcher.h"
#include "Util.h"
#include "Parameters.h"

#include <iostream>
#include <algorithm>
#include <limits>
#include <cfloat>
#include <numeric>
#include <cmath>
#include <vector>


#ifdef OPENMP
#include <omp.h>
#endif


FwBwAligner::FwBwAligner(size_t queryLen, size_t targetLen, size_t length, size_t blocks,
                         SubstitutionMatrix &subMat){

    zmForward = malloc_matrix<float>(queryLen, targetLen);
    zeForward = malloc_matrix<float>(queryLen, targetLen);
    zfForward = malloc_matrix<float>(queryLen, targetLen);
    zmBackward = malloc_matrix<float>(queryLen, targetLen);
    zeBackward = malloc_matrix<float>(queryLen, targetLen);
    zfBackward = malloc_matrix<float>(queryLen, targetLen);
    scoreForward = malloc_matrix<float>(queryLen, targetLen);
    scoreBackward = malloc_matrix<float>(queryLen, targetLen);
    P = malloc_matrix<float>(queryLen, targetLen);

    // Define block matrices
    // int length = targetLen / 4;
    vj = static_cast<float*>(mem_align(16, length * sizeof(float)));
    wj = static_cast<float*>(mem_align(16, length * sizeof(float)));

    // Matrices used outside forwardBackwardSaveBlockMaxLocal, so shape is blocks, queryLen
    zmaxBlocksMaxForward = malloc_matrix<float>(blocks, queryLen);
    zmaxBlocksMaxBackward = malloc_matrix<float>(blocks, queryLen);

    // Matrices used inside forwardBackwardSaveBlockMaxLocal, so shape is queryLen + 1
    zmaxForward = static_cast<float*>(malloc((queryLen + 1) * sizeof(float)));
    memset(zmaxForward, 0, (queryLen + 1) * sizeof(float)); 
    zmaxBackward = static_cast<float*>(malloc((queryLen + 1) * sizeof(float)));
    memset(zmaxBackward, 0, (queryLen + 1) * sizeof(float)); 


    // mat3di = malloc_matrix<float>(21, 21);
    blosum = malloc_matrix<float>(21, 21);
    for (int i = 0; i < subMat.alphabetSize; ++i) {
        for (int j = 0; j < subMat.alphabetSize; ++j) {
            blosum[i][j] = static_cast<float>(subMat.subMatrix[i][j]) * 2;
        }
    }
    //Debug: print blosum
    // std::cout << "blosum" << std::endl;
    // for (int i = 0; i < 21; ++i) {
    //     for (int j = 0; j < 21; ++j) {
    //         std::cout << blosum[i][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
}


void FwBwAligner::computeForwardScoreMatrix(const unsigned char* queryNum, const unsigned char* targetNum,
                                                unsigned int queryLen, unsigned int targetLen,
                                                float** blosum, float T, float ** scoreForward) {
    for (size_t i = 0; i < queryLen; ++i) {
        for (size_t j = 0; j < targetLen; ++j) {
            scoreForward[i][j] = blosum[queryNum[i]][targetNum[j]];
            scoreForward[i][j] = exp(scoreForward[i][j] / T);
        }
    }
}


FwBwAligner::~FwBwAligner(){
    free(scoreForward);
    free(scoreBackward);
    free(zmForward);
    free(zeForward);
    free(zfForward);
    free(zmBackward);
    free(zeBackward);
    free(zfBackward);
    free(P);
    free(zmaxBlocksMaxForward);
    free(zmaxBlocksMaxBackward);
    free(zmaxForward);
    free(zmaxBackward);
    free(vj);
    free(wj);
    free(blosum);
}


FwBwAligner::s_align FwBwAligner::align(const std::string & querySeq, const std::string & targetSeq, size_t queryLen, size_t targetLen,
                                            size_t length, size_t blocks, SubstitutionMatrix &subMat){

    unsigned char* queryNum = seq2num(querySeq, subMat.aa2num);
    unsigned char* targetNum = seq2num(targetSeq, subMat.aa2num);


    // Debug: Print queryNum and targetNum
    // std::cout << "queryNum and targetNum" << std::endl;
    // for (size_t i = 0; i < queryLen; ++i) {
    //     std::cout << static_cast<int>(queryNum[i]) << " ";
    // }
    // std::cout << std::endl;
    // for (size_t i = 0; i < targetLen; ++i) {
    //     std::cout << static_cast<int>(targetNum[i]) << " ";
    // }
    // std::cout << std::endl;

    const float T = 10;
    const float go = -3.5;
    const float ge = -0.3;
    computeForwardScoreMatrix(queryNum, targetNum,
                              queryLen, targetLen,
                              blosum, T, scoreForward);
    for (size_t i = 0; i < queryLen; ++i) {
        for (size_t j = 0; j < targetLen; ++j) {
            scoreBackward[i][j] = scoreForward[queryLen - 1 - i][targetLen - 1 - j];
        }
    }
    // Debug: print scoreforward
    // std::cout << "scoreForward: " << std::endl;
    // for (size_t i = 0; i < queryLen; ++i) {
    //     for (size_t j = 0; j < targetLen; ++j) {
    //         std::cout << scoreForward[i][j] << " ";
    //     }

    //     std::cout << std::endl;
    // }
    // Debug: print scorebackward
    // std::cout << "scoreBackward" << std::endl;
    // for (size_t i = 0; i < queryLen; ++i) {
    //     for (size_t j = 0; j < targetLen; ++j) {
    //         std::cout << scoreBackward[i][j] << " ";
    //     }
    // }
    //     std::cout << std::endl;


    // size_t length = targetLen / 4;
    std::cout << "queryLen: " << queryLen << " targetLen: " << targetLen << " length: " << length << " blocks: " << blocks << std::endl;
    for (size_t i = 0; i < length; ++i) {
        vj[i] = exp(((length - 1) * ge + go - i * ge) / T);
        wj[i] = exp(((length - 1) * ge - i * ge) / T);
    }

    for (size_t i = 0; i < queryLen; ++i) {
        for (size_t j = 0; j < targetLen; ++j) {
            zmForward[i][j] = -DBL_MAX;
            zeForward[i][j] = -DBL_MAX;
            zfForward[i][j] = -DBL_MAX;
            zmBackward[i][j] = -DBL_MAX;
            zeBackward[i][j] = -DBL_MAX;
            zfBackward[i][j] = -DBL_MAX;
        }
    }

    //initialize zInit
    float* zInitForward[3];
    zInitForward[0] = new float[queryLen];
    zInitForward[1] = new float[queryLen];
    zInitForward[2] = new float[queryLen];

    for (unsigned int i=0; i < queryLen; ++i){
        zInitForward[0][i] = zmForward[i][0];
        zInitForward[1][i] = zeForward[i][0];
        zInitForward[2][i] = zfForward[i][0];
    }
    // float** zmaxBlocksMaxForward = malloc_matrix<float>(blocks + 1, queryLen + 1);
    // std::cout << "Forward start" << std::endl;
    for (size_t b = 0; b < blocks; ++b) {
        size_t start = b * length;
        size_t end = (b + 1) * length;
        // std::cout << "Block: " << b << " start " << start << " end " << end << std::endl;
        // number of cols to memcpy in forwardBackwardSaveBlockMaxLocal
        size_t memcpy_cols = std::min(end, targetLen) - start;
        forwardBackwardSaveBlockMaxLocal(scoreForward, zInitForward, vj, wj, T, go, ge, queryLen, start, end, memcpy_cols,
                                         zmForward, zeForward, zfForward,
                                         zmaxForward);
        
        memcpy(zmaxBlocksMaxForward[b], zmaxForward, queryLen * sizeof(float));
    }

    ///////////////////////////////////Backward////////////////////////////////////////
    
    float* zInitBackward[3];
    zInitBackward[0] = new float[queryLen];
    zInitBackward[1] = new float[queryLen];
    zInitBackward[2] = new float[queryLen];

    for (unsigned int i=0; i < queryLen; ++i){
        zInitBackward[0][i] = zmBackward[i][0];
        zInitBackward[1][i] = zeBackward[i][0];
        zInitBackward[2][i] = zfBackward[i][0];
    }

    // float** zmaxBlocksMaxBackward = malloc_matrix<float>(blocks+1, queryLen + 1);
    // memcpy(zmaxBlocksMaxBackward[0], zmaxBackward, (queryLen + 1) * sizeof(float));
    // std::cout << "Backward start" << std::endl;
    for (size_t b = 0; b < blocks; ++b) {
        // std::cout << "Block " << b << std::endl;
        size_t start = b * length;
        size_t end = (b + 1) * length;
        size_t memcpy_cols = std::min(end, targetLen) - start;
        
        forwardBackwardSaveBlockMaxLocal(scoreBackward, zInitBackward, vj, wj, T, go, ge, queryLen, start, end, memcpy_cols,
                                         zmBackward, zeBackward, zfBackward,
                                         zmaxBackward);
        memcpy(zmaxBlocksMaxBackward[b], zmaxBackward, queryLen * sizeof(float));
    }

    ///////////////////////////////////Rescale////////////////////////////////////////
    // Rescale the values by the maximum in the log space for each block
    // This turns the matrix into log space
    // Debug:: print zmForward before rescaling
    std::cout << "zmForward before rescaling: " << std::endl;
    for (size_t i = 0; i < queryLen; ++i) {
        for (size_t j = 0; j < targetLen; ++j) {
            std::cout << zmForward[i][j] << " ";
        }
        std::cout << std::endl;
    }
    // Debug: print zmaxBlocksMaxForward
    std::cout << "zmaxBlocksMaxForward: " << std::endl;
    // print number of rows and columns of zmaxBlocksMaxForward
    std::cout << "rows: " << blocks << " columns: " << queryLen << std::endl;

    for (size_t i = 0; i < blocks; ++i) {
        for (size_t j = 0; j < queryLen; ++j) {
            std::cout << zmaxBlocksMaxForward[i][j] << " ";
        }
        std::cout << std::endl;
    }
    rescaleBlocks(zmForward, zmaxBlocksMaxForward, queryLen, length, blocks, targetLen);
    rescaleBlocks(zmBackward, zmaxBlocksMaxBackward, queryLen, length, blocks, targetLen);
    // Debug:: print zmForward
    std::cout << "zmForward after rescaling: " << std::endl;
    for (size_t i = 0; i < queryLen; ++i) {
        for (size_t j = 0; j < targetLen; ++j) {
            std::cout << zmForward[i][j] << " ";
        }
        std::cout << std::endl;
    }

   // compute zm max
    float max_zm = -DBL_MAX;
    for (size_t i = 0; i < queryLen; ++i) {
        for (size_t j = 0; j < targetLen; ++j) {
            max_zm = std::max(max_zm, zmForward[i][j]);
        }
    }
    // std::cout << "max zm\t" << max_zm << "\n";
    // Debug logsumexp_zm
    // float ze_11 = zeForward[queryLen - 1][targetLen - 1] + max_zm;
    // float zf_11 = zfForward[queryLen - 1][targetLen - 1] + max_zm;
    // float max_val = std::max({max_zm, ze_11, zf_11});
    // float logsumexp_zm_mine = max_val + log(exp(max_zm - max_val) + exp(ze_11 - max_val) + exp(zf_11 - max_val));

    // compute sum_exp
    float sum_exp= 0.0;
    for (size_t i = 0; i < queryLen; ++i) {
        for (size_t j = 0; j < targetLen; ++j) {
            sum_exp += exp(zmForward[i][j] - max_zm);
        }
    }
    float logsumexp_zm = max_zm + log(sum_exp);
    // std::cout << "logsumexp_zm\t" << logsumexp_zm << " " << logsumexp_zm_mine << "\n";
    // compute posterior probabilities
    float max_p = 0.0;
    size_t max_i;
    size_t max_j;
    for (size_t i = 0; i < queryLen; ++i) {
        for (size_t j = 0; j < targetLen; ++j) {
            P[i][j] = exp(
                zmForward[i][j]
                + zmBackward[queryLen - 1 - i][targetLen - 1 - j]
                - log(scoreForward[i][j]) // FIXME scoreForward is already exp(S/T)
                - logsumexp_zm
            );
            if (P[i][j] > max_p) {
                max_p = P[i][j];
                max_i = i;
                max_j = j;
            }
        }
    }
    // Print elements of P[0][0]
    std::cout << "P[0][0]: " << P[0][0] << " " << zmForward[0][0] << " " << zmBackward[queryLen - 1][targetLen - 1] << " " << log(scoreForward[0][0]) << " " << logsumexp_zm << "\n";
    //Debug: print querySeq and targetSeq
    // std::cout << "querySeq:\n" << querySeq << "\ntargetSeq:\n" << targetSeq << "\n";
    //Debug: print P
    // for (size_t i = 0; i < queryLen; ++i) {
    //     for (size_t j = 0; j < targetLen; ++j) {
    //         std::cout << P[i][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // print elements of P[max_i][max_j]
    Debug(Debug::INFO) << "Index: " << max_i << " " << max_j << " elements: " << zmForward[max_i][max_j] << " " << zmBackward[queryLen - 1 - max_i][targetLen - 1 - max_j] << " " << log(scoreForward[max_i][max_j]) << " " << logsumexp_zm << " " << max_p << "\n";
    // If max_p is above 100000, print max_p, querySeq, targetSeq and terminate
    if (max_p > 1.0 || max_p < 0.0) {
        Debug(Debug::ERROR) << "Invalid maxprob.\n";
        for (size_t i = 0; i < queryLen; ++i) {
            for (size_t j = 0; j < targetLen; ++j) {
                std::cout << P[i][j] << " ";
            }
        std::cout << std::endl;
        }
        std::cout << "error maxprob: " << max_p << "\nquerySeq:\n" << querySeq << "\ntargetSeq:\n" << targetSeq << "\n";
        // size_t count = 0;
        // while (querySeq[count] != '\0') {
        //     if (querySeq[count] == '\n') {
        //         std::cout << "newline\n" << count << "\n";
        //     }
        //     else {
        //         count++;
        //     }
        // }
        // std::cout << "querySeq length: " << count << "\n";
        EXIT(EXIT_FAILURE);
    }
    // traceback 
    s_align result;
    result.cigar = "";
    result.cigar.reserve(queryLen + targetLen);
    result.maxProb = max_p;
    result.qEndPos1 = max_i;
    result.dbEndPos1 = max_j;
    float d;
    float l;
    float u;
    while (max_i > 0 && max_j > 0) {
        d = P[max_i - 1][max_j - 1];
        l = P[max_i][max_j - 1];
        u = P[max_i - 1][max_j];
        // std::cout << std::fixed << std::setprecision(8) << max_i << '\t' << max_j << '\t' << d << '\t' << l << '\t' << u << '\n';
        if (d > l && d > u) {
            max_i--;
            max_j--;
            result.cigar.push_back('M');
        } else if (l > d && l > u) {
            max_j--;
            result.cigar.push_back('I');
        } else {
            max_i--;
            result.cigar.push_back('D');
        }
    }
    result.qStartPos1 = max_i;
    result.dbStartPos1 = max_j;
    result.cigarLen = result.cigar.length();
    std::reverse(result.cigar.begin(), result.cigar.end());

    free(queryNum);
    free(targetNum);
    delete[] zInitForward[0];
    delete[] zInitForward[1];
    delete[] zInitForward[2];
    delete[] zInitBackward[0];
    delete[] zInitBackward[1];
    delete[] zInitBackward[2];

    return result;
}

void FwBwAligner::forwardBackwardSaveBlockMaxLocal(float** S, float** z_init,
                                                   float* vj, float* wj,
                                                   float T, float go, float ge,
                                                   size_t rows, size_t start, size_t end, size_t memcpy_cols,
                                                   float** zm, float** ze, float** zf, float* zmax) {
    float exp_go = exp(go / T);
    float exp_ge = exp(ge / T);
    
    float** zmBlock = malloc_matrix<float>(rows + 1, end - start + 1);
    float** zeBlock = malloc_matrix<float>(rows + 1, end - start + 1);
    float** zfBlock = malloc_matrix<float>(rows + 1, end - start + 1);
    
    //Init blocks
    memset(zmBlock[0], 0, (end - start + 1) * sizeof(float));
    memset(zeBlock[0], 0, (end - start + 1) * sizeof(float));
    memset(zfBlock[0], 0, (end - start + 1) * sizeof(float));


    // Initialize the first column of the segment starting from the second row
    for (size_t i = 0; i < rows; ++i) {
        zmBlock[i+1][0] = z_init[0][i];
        zeBlock[i+1][0] = z_init[1][i];
        zfBlock[i+1][0] = z_init[2][i];
    }

    size_t cols = end - start;
    float* exp_ge_arr = static_cast<float*>(mem_align(16, cols * sizeof(float)));
    for (size_t i = 0; i < cols; ++i) {
        exp_ge_arr[i] = exp((i * ge + ge) / T);
    }

    float current_max = 0;
    for (size_t i = 1; i <= rows; ++i) {
        if (i != 1) {
            zmBlock[i - 1][0] = exp(zmBlock[i - 1][0]);
            zeBlock[i - 1][0] = exp(zeBlock[i - 1][0]);
            zfBlock[i - 1][0] = exp(zfBlock[i - 1][0]);
        }
        for (size_t j = 1; j <= cols; ++j) {
            float tmp = (zmBlock[i - 1][j - 1] + zeBlock[i - 1][j - 1] + zfBlock[i - 1][j - 1] + exp(-current_max));
            zmBlock[i][j] = tmp * S[i - 1][start + j - 1];
        }
        
        float* zm_exp = static_cast<float*>(mem_align(16, cols * sizeof(float)));
        memcpy(zm_exp, zmBlock[i], cols * sizeof(float));
        zm_exp[0] = exp(zm_exp[0]);

        // Correct translation of the cumulative sum
        float cumulative_sum = 0;
        for (size_t j = 1; j <= cols; ++j) {
            cumulative_sum += zm_exp[j - 1] * vj[j - 1];
            zeBlock[i][j] = (cumulative_sum / wj[j - 1]) + exp(zeBlock[i][0]) * exp_ge_arr[j - 1];
        }
        for (size_t j = 1; j <= cols; ++j) {
            zfBlock[i][j] = (zmBlock[i - 1][j] * exp_go + zfBlock[i - 1][j] * exp_ge);
        }


        float z_temp = *std::max_element(zmBlock[i] + 1, zmBlock[i] + cols + 1);
        zmax[i-1] = log(z_temp);
        current_max += zmax[i-1];
        for (size_t j = 1; j <= cols; ++j) {
            zmBlock[i][j] /= z_temp;
            zeBlock[i][j] /= z_temp;
            zfBlock[i][j] /= z_temp;
        }

        for (size_t j = i; j <= rows; ++j) {
            zmBlock[j][0] -= zmax[i-1];
            zeBlock[j][0] -= zmax[i-1];
            zfBlock[j][0] -= zmax[i-1];
        }

        free(zm_exp);
    }

    //Calculate the cumulative sum of zmax[1:]
    std::vector<float> rescale(rows);
    // std::partial_sum(zmax + 1, zmax + rows + 1, rescale.begin());
    std::partial_sum(zmax, zmax + rows, rescale.begin());

    //Fixme
    // 
    for (size_t i = 0; i < rows; ++i) {
        z_init[0][i] = log(zmBlock[i + 1][memcpy_cols]) + rescale[i-1];
        z_init[1][i] = log(zeBlock[i + 1][memcpy_cols]) + rescale[i-1];
        z_init[2][i] = log(zfBlock[i + 1][memcpy_cols]) + rescale[i-1];
    }

    for (size_t i = 0; i < rows; ++i) {
        memcpy(zm[i] + start, zmBlock[i+1]+1, memcpy_cols * sizeof(float));
        memcpy(ze[i] + start, zeBlock[i+1]+1, memcpy_cols * sizeof(float));
        memcpy(zf[i] + start, zfBlock[i+1]+1, memcpy_cols * sizeof(float));
    }

    free(zmBlock);
    free(zeBlock);
    free(zfBlock);
    free(exp_ge_arr);
}

void FwBwAligner::rescaleBlocks(float **matrix, float **scale, size_t rows, size_t length, size_t blocks, size_t targetLen){
    // Function to rescale the values by the maximum in the log space for each block
    for (size_t b = 0; b < blocks; ++b) {
        size_t start = b * length;
        size_t end = std::min((b + 1) * length, targetLen);
        // size_t end = (b + 1) * length;
        std::vector<float> cumsum(rows);
        std::partial_sum(scale[b], scale[b] + rows, cumsum.begin());
        // print cumsum vector for each block
        std::cout << "block " << b << " cumsum: ";
        for (size_t i = 0; i < rows; ++i) {
            std::cout << cumsum[i] << " ";
        }
        std::cout << std::endl;

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = start; j < end; ++j) {
                // matrix[i][j] = log(matrix[i][j]) ;// + cumsum[i-1];
                matrix[i][j] = log(matrix[i][j]) + cumsum[i];
            }
        }
    }
}

int fwbw(int argc, const char **argv, const Command &command) {
    //Prepare the parameters & DB
    Parameters &par = Parameters::getInstance();
    par.parseParameters(argc, argv, command, true, 0, MMseqsParameter::COMMAND_ALIGN);
    DBReader<unsigned int> qdbr(par.db1.c_str(), par.db1Index.c_str(), par.threads, DBReader<unsigned int>::USE_DATA | DBReader<unsigned int>::USE_INDEX);
    qdbr.open(DBReader<unsigned int>::NOSORT);
    DBReader<unsigned int> tdbr(par.db2.c_str(), par.db2Index.c_str(), par.threads, DBReader<unsigned int>::USE_DATA | DBReader<unsigned int>::USE_INDEX);
    tdbr.open(DBReader<unsigned int>::NOSORT);
    DBReader<unsigned int> alnRes (par.db3.c_str(), par.db3Index.c_str(), par.threads, DBReader<unsigned int>::USE_DATA | DBReader<unsigned int>::USE_INDEX);
    alnRes.open(DBReader<unsigned int>::LINEAR_ACCCESS);

    DBWriter fwbwAlnWriter(par.db4.c_str(), par.db4Index.c_str(), par.threads, par.compressed, Parameters::DBTYPE_ALIGNMENT_RES);
    fwbwAlnWriter.open();

    // const size_t targetColumn = (par.targetTsvColumn == 0) ? SIZE_T_MAX :  par.targetTsvColumn - 1;

    // Initialize the alignment mode
    // float gapOpen, gapExtend, Temperature;
    const int querySeqType = qdbr.getDbtype();
    if (Parameters::isEqualDbtype(querySeqType, Parameters::DBTYPE_NUCLEOTIDES)) {
        // m = new NucleotideMatrix(par.scoringMatrixFile.values.nucleotide().c_str(), 1.0, par.scoreBias);
        // gapOpen = par.gapOpen.values.nucleotide();
        // gapExtend = par.gapExtend.values.nucleotide();
        Debug(Debug::ERROR) << "Invalid datatype. Nucleotide.\n";
        EXIT(EXIT_FAILURE);
    } 
    SubstitutionMatrix subMat = SubstitutionMatrix(par.scoringMatrixFile.values.aminoacid().c_str(), 1.4, par.scoreBias); // Check : par.scoreBias = 0.0
        // gapOpen = -3.5;
        // gapExtend = -0.3;
        // Temperature = 10;

    const size_t flushSize = 100000000;
    size_t iterations = static_cast<int>(ceil(static_cast<double>(alnRes.getSize()) / static_cast<double>(flushSize)));
    Debug(Debug::INFO) << "Processing " << iterations << " iterations\n";
    for (size_t i = 0; i < iterations; i++) {
        size_t start = (i * flushSize);
        size_t bucketSize = std::min(alnRes.getSize() - (i * flushSize), flushSize);
        Debug::Progress progress(bucketSize);

#pragma omp parallel
        {
            unsigned int thread_idx = 0;

#ifdef OPENMP
            thread_idx = (unsigned int) omp_get_thread_num();
#endif
            const char *entry[255];
            // std::string alnResultsOutString;
            // alnResultsOutString.reserve(1024*1024);
            std::string alnResultsOutString;
            char buffer[1024 + 32768*4];
            std::vector<Matcher::result_t> alnResults;
            alnResults.reserve(300);
            std::vector<Matcher::result_t> localFwbwResults;
#pragma omp for schedule(dynamic,1)
            for (size_t id = start; id < (start + bucketSize); id++) {
                progress.updateProgress();
                unsigned int key = alnRes.getDbKey(id);
                Debug(Debug::INFO) << "start " << id << "\n";
                const unsigned int queryId = qdbr.getId(key);
                const unsigned int queryKey = qdbr.getDbKey(queryId);
                char *alnData = alnRes.getData(id, thread_idx);
                alnResults.clear();
                localFwbwResults.clear();
                while (*alnData!='\0'){
                    const size_t columns = Util::getWordsOfLine(alnData, entry, 255);
                    if (columns >= Matcher::ALN_RES_WITHOUT_BT_COL_CNT) {
                        alnResults.emplace_back(Matcher::parseAlignmentRecord(alnData, true));                        
                    } else {
                        Debug(Debug::ERROR) << "Invalid input result format ("<<columns<<" columns).\n";
                        EXIT(EXIT_FAILURE);
                    }
                    alnData = Util::skipLine(alnData);
                }
                if (alnResults.size() == 0){
                    continue;
                }

                // FIXME:: Temporary way to fix the issue of the last newline character in the sequence
                const char* originalQuerySeq = qdbr.getData(queryKey, thread_idx);
                size_t qlen = strlen(originalQuerySeq); 
                char* querySeq = new char[qlen];
                if (qlen > 0 && originalQuerySeq[qlen - 1] == '\n') {
                    strncpy(querySeq, originalQuerySeq, qlen - 1);  // Copy all but the last character
                    querySeq[qlen - 1] = '\0';  // Null-terminate the new string
                } else {
                    strcpy(querySeq, originalQuerySeq);  // Copy the entire string if no newline
                }
                size_t queryLen = strlen(querySeq);


                fwbwAlnWriter.writeStart(thread_idx);
                // char * tmpBuff = Itoa::u32toa_sse2((uint32_t) queryKey, buffer);
                // *(tmpBuff-1) = '\t';
                // const unsigned int queryIdLen = tmpBuff - buffer;
                for (size_t i=0; i < alnResults.size(); i++){
                    unsigned int targetKey = alnResults[i].dbKey;

                    // FIXME:: Temporary way to fix the issue of the last newline character in the sequence
                    const char* originalTargetSeq = tdbr.getData(targetKey, thread_idx);
                    size_t len = strlen(originalTargetSeq); 
                    char* targetSeq = new char[len];
                    if (len > 0 && originalTargetSeq[len - 1] == '\n') {
                        strncpy(targetSeq, originalTargetSeq, len - 1);  // Copy all but the last character
                        targetSeq[len - 1] = '\0';  // Null-terminate the new string
                    } else {
                        strcpy(targetSeq, originalTargetSeq);  // Copy the entire string if no newline
                    }
                    size_t targetLen = strlen(targetSeq);


                    size_t length = 16;
                    size_t blocks = (targetLen / length) + (targetLen % length != 0);
                    FwBwAligner fwbwAligner(queryLen, targetLen, length, blocks, subMat);
                    FwBwAligner::s_align fwbwAlignment = fwbwAligner.align(querySeq, targetSeq, queryLen, targetLen, length, blocks, subMat);
                    // initialize values of result_t
                    float qcov = 0.0;
                    float dbcov = 0.0;
                    float seqId = 0.0;
                    float evalue = fwbwAlignment.maxProb;

                    const unsigned int alnLength = fwbwAlignment.cigarLen;
                    const int score = 0;
                    const unsigned int qStartPos = fwbwAlignment.qStartPos1;
                    const unsigned int dbStartPos = fwbwAlignment.dbStartPos1;
                    const unsigned int qEndPos = fwbwAlignment.qEndPos1;
                    const unsigned int dbEndPos = fwbwAlignment.dbEndPos1;
                    std::string backtrace = fwbwAlignment.cigar;

                    // Map s_align values to result_t
                    Matcher::result_t res = Matcher::result_t(targetKey, score, qcov, dbcov, seqId, evalue, alnLength, qStartPos, qEndPos, queryLen, dbEndPos, dbStartPos, targetLen, backtrace);
                    
                    localFwbwResults.emplace_back(res);

                    // FIXME: will not be needed after the newline character issue is fixed
                    delete[] targetSeq;
                }

                // sort local results. They will currently be sorted by first fwbwscore, then targetlen, then by targetkey.
                SORT_SERIAL(localFwbwResults.begin(), localFwbwResults.end(), Matcher::compareHits);
                std::vector<Matcher::result_t> *returnRes = &localFwbwResults;
                for (size_t result = 0; result < returnRes->size(); result++) {
                    size_t len = Matcher::resultToBuffer(buffer, (*returnRes)[result], par.addBacktrace);
                    alnResultsOutString.append(buffer, len);
                }
                // Debug(Debug::INFO) << "debug " << alnResultsOutString.c_str() << " ";
                fwbwAlnWriter.writeData(alnResultsOutString.c_str(), alnResultsOutString.length(), queryKey, thread_idx);
                alnResultsOutString.clear();
                // for (size_t i=0; i < localFwbwResults.size(); i++){
                //     char* basePos = tmpBuff;
                //     tmpBuff = Util::fastSeqIdToBuffer(localFwbwResults[i].first, tmpBuff);
                //     *(tmpBuff-1) = '\t';
                //     const unsigned int probLen = tmpBuff - basePos;
                //     size_t alnLen = Matcher::resultToBuffer(tmpBuff, localFwbwResults[i].second, par.addBacktrace);
                //     fwbwAlnWriter.writeAdd(buffer, queryIdLen+probLen+alnLen, thread_idx);
                // }
                // fwbwAlnWriter.writeEnd(queryKey, thread_idx);
                alnResults.clear();
                localFwbwResults.clear();
                Debug(Debug::INFO) << "end " << id << "\n";
                
                // FIXME: will not be needed after the newline character issue is fixed
                delete[] querySeq;
            }
        }
        alnRes.remapData();
    }
    Debug(Debug::INFO) << "All Done\n";
    fwbwAlnWriter.close();
    alnRes.close();
    qdbr.close();
    tdbr.close();
    return EXIT_SUCCESS;

}