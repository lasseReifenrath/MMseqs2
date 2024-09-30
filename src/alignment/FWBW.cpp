#include "FWBW.h"
#include "Debug.h"
#include "DBReader.h"
#include "DBWriter.h"

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

// FwBwAligner::FwBwAligner(const std::string &querySeqDB, const std::string &targetSeqDB,
//                          const std::string &alnDB, 
//                          const std::string &outDB, const std::string &outDBIndex, const Parameters &par){
    
    
//     // DBReader<unsigned int> qdbr(par.db1.c_str(), par.threads, DBReader<unsigned int>::USE_DATA | DBReader<unsigned int>::USE_INDEX);
//     // qdbr.open(DBReader<unsigned int>::NOSORT);
//     // DBReader<unsigned int> tdbr(par.db2.c_str(), par.threads, DBReader<unsigned int>::USE_DATA | DBReader<unsigned int>::USE_INDEX);
//     // tdbr.open(DBReader<unsigned int>::NOSORT);
//     // DBReader<unsigned int> alnRes (par.db3.c_str(), par.threads, DBReader<unsigned int>::USE_DATA | DBReader<unsigned int>::USE_INDEX);
//     // alnRes.open(DBReader<unsigned int>::NOSORT);

//     // DBWriter fwbwAlnRes(par.db4.c_str(), par.db4Index.c_str(), par.threads, par.compressed, Parameters::DBTYPE_ALIGNMENT_RES);
//     // fwbwAlnRes.open();
//     tdbr = tDbrIdx->sequenceReader;
//     targetSeqType = tdbr->getDbtype();
//     sameQTDB = (targetSeqDB.compare(querySeqDB) == 0);

//     if (querySeqType == -1 || targetSeqType == -1) {
//         Debug(Debug::ERROR) << "Please recreate your database or add a .dbtype file to your sequence/profile database.\n";
//         EXIT(EXIT_FAILURE);
//     }

//     alnResDB = new DBReader<unsigned int>(alnDB.c_str(), par.threads, DBReader<unsigned int>::USE_DATA | DBReader<unsigned int>::USE_INDEX);
//     alnResDB->open(DBReader<unsigned int>::NOSORT);

//     if (Parameters::isEqualDbtype(querySeqType, Parameters::DBTYPE_NUCLEOTIDES)) {
//         // m = new NucleotideMatrix(par.scoringMatrixFile.values.nucleotide().c_str(), 1.0, par.scoreBias);
//         // gapOpen = par.gapOpen.values.nucleotide();
//         // gapExtend = par.gapExtend.values.nucleotide();
//         Debug(Debug::ERROR) << "Invalid datatype. Nucleotide.\n";
//         EXIT(EXIT_FAILURE);
//     } else{
//         subMat = new SubstitutionMatrix(par.scoringMatrixFile.values.aminoacid().c_str(), 1.4, par.scoreBias); // Check : par.scoreBias = 0.0
//         gapOpen = -3.5;
//         gapExtend = -0.3;
//         Temperature = 10;
        
//     }

//     zmForward = malloc_matrix<float>(maxQueryLen, maxTargetLen);
//     zeForward = malloc_matrix<float>(maxQueryLen, maxTargetLen);
//     zfForward = malloc_matrix<float>(maxQueryLen, maxTargetLen);
//     zmBackward = malloc_matrix<float>(maxQueryLen, maxTargetLen);
//     zeBackward = malloc_matrix<float>(maxQueryLen, maxTargetLen);
//     zfBackward = malloc_matrix<float>(maxQueryLen, maxTargetLen);
//     scoreForward = malloc_matrix<float>(maxQueryLen, maxTargetLen);
//     scoreBackward = malloc_matrix<float>(maxQueryLen, maxTargetLen);
//     P = malloc_matrix<float>(maxQueryLen, maxTargetLen);
//     int length = maxTargetLen / 4;
//     vj = static_cast<float*>(mem_align(16, length * sizeof(float)));
//     wj = static_cast<float*>(mem_align(16, length * sizeof(float)));

//     int blocks = maxTargetLen / length;

//     zmaxBlocksMaxForward = malloc_matrix<float>(blocks, maxQueryLen + 1);
//     zmaxBlocksMaxBackward = malloc_matrix<float>(blocks, maxQueryLen + 1);
//     zmaxForward = static_cast<float*>(malloc((maxQueryLen + 1) * sizeof(float)));
//     memset(zmaxForward, 0, (maxQueryLen + 1) * sizeof(float)); 
//     zmaxBackward = static_cast<float*>(malloc((maxQueryLen + 1) * sizeof(float)));
//     memset(zmaxBackward, 0, (maxQueryLen + 1) * sizeof(float)); 


//     mat3di = malloc_matrix<float>(21, 21);
//     blosum = malloc_matrix<float>(21, 21);
//     for (size_t i = 0; i < subMat.alphabetSize; ++i) {
//         for (size_t j = 0; j < subMat.alphabetSize; ++j) {
//             mat3di[i][j] = static_cast<float>(subMat3Di.subMatrix[i][j]) * 2;
//             blosum[i][j] = static_cast<float>(subMat.subMatrix[i][j]) * 2;
//         }
//     }
// }


// void FwBwAligner::computeForwardScoreMatrix(const unsigned char* ss1_num, const unsigned char* ss2_num,
//                                                 const unsigned char* aa1_num, const unsigned char* aa2_num,
//                                                 unsigned int queryLen, unsigned int targetLen,
//                                                 float** mat3di, float** blosum, float T, float ** scoreForward) {
//     for (size_t i = 0; i < queryLen; ++i) {
//         for (size_t j = 0; j < targetLen; ++j) {
//             scoreForward[i][j] = mat3di[ss1_num[i]][ss2_num[j]] + blosum[aa1_num[i]][aa2_num[j]];
//             scoreForward[i][j] = exp(scoreForward[i][j] / T);
//         }
//     }
// }

// FwBwAligner::~FwBwAligner(){
//     free(scoreForward);
//     free(scoreBackward);
//     free(zmForward);
//     free(zeForward);
//     free(zfForward);
//     free(zmBackward);
//     free(zeBackward);
//     free(zfBackward);
//     free(P);
// //    free_aligned_matrix(P);
//     free(zmaxBlocksMaxForward);
//     free(zmaxBlocksMaxBackward);
//     free(zmaxForward);
//     free(zmaxBackward);
//     free(vj);
//     free(wj);
// }


// FwBwAligner::s_align FwBwAligner::align(std::string & query3Di, std::string & queryAA,
//                                         std::string & target3Di, std::string & targetAA){
//     const unsigned int queryLen = query3Di.size();
//     const unsigned int targetLen = target3Di.size();

//     unsigned char* ss1_num = seq2num(query3Di, subMat3Di.aa2num);
//     unsigned char* ss2_num = seq2num(target3Di, subMat3Di.aa2num);
//     unsigned char* aa1_num = seq2num(queryAA, subMat.aa2num);
//     unsigned char* aa2_num = seq2num(targetAA, subMat.aa2num);

//     const float T = 10;
//     const float go = -3.5;
//     const float ge = -0.3;
//     computeForwardScoreMatrix(ss1_num, ss2_num, aa1_num, aa2_num,
//                               queryLen, targetLen, mat3di, blosum, T, scoreForward);
//     for (size_t i = 0; i < queryLen; ++i) {
//         for (size_t j = 0; j < targetLen; ++j) {
//             scoreBackward[i][j] = scoreForward[queryLen - 1 - i][targetLen - 1 - j];
//         }
//     }

//     int length = targetLen / 4;
//     int blocks = targetLen / length;

//     for (size_t i = 0; i < length; ++i) {
//         vj[i] = exp(((length - 1) * ge + go - i * ge) / T);
//         wj[i] = exp(((length - 1) * ge - i * ge) / T);
//     }

//     for (size_t i = 0; i < queryLen; ++i) {
//         for (size_t j = 0; j < targetLen; ++j) {
//             zmForward[i][j] = -DBL_MAX;
//             zeForward[i][j] = -DBL_MAX;
//             zfForward[i][j] = -DBL_MAX;
//             zmBackward[i][j] = -DBL_MAX;
//             zeBackward[i][j] = -DBL_MAX;
//             zfBackward[i][j] = -DBL_MAX;
//         }
//     }

//     //initialize zInit
//     float* zInitForward[3];
//     zInitForward[0] = new float[queryLen];
//     zInitForward[1] = new float[queryLen];
//     zInitForward[2] = new float[queryLen];

//     for (unsigned int i=0; i < queryLen; ++i){
//         zInitForward[0][i] = zmForward[i][0];
//         zInitForward[1][i] = zeForward[i][0];
//         zInitForward[2][i] = zfForward[i][0];
//     }
//     float** zmaxBlocksMaxForward = malloc_matrix<float>(blocks+1, queryLen + 1);
//     // memcpy(zmaxBlocksMaxForward[0], zmaxForward, (queryLen + 1) * sizeof(float));
//     for (int b = 0; b < blocks; ++b) {
//         int start = b * length;
//         int end = (b + 1) * length;
//         forwardBackwardSaveBlockMaxLocal(scoreForward, zInitForward, vj, wj, T, go, ge, queryLen, start, end,
//                                          zmForward, zeForward, zfForward,
//                                          zmaxForward);
        
//         memcpy(zmaxBlocksMaxForward[b+1], zmaxForward, (queryLen + 1) * sizeof(float));
//     }

//     ///////////////////////////////////Backward////////////////////////////////////////
    
//     float* zInitBackward[3];
//     zInitBackward[0] = new float[queryLen];
//     zInitBackward[1] = new float[queryLen];
//     zInitBackward[2] = new float[queryLen];

//     for (unsigned int i=0; i < queryLen; ++i){
//         zInitBackward[0][i] = zmBackward[i][0];
//         zInitBackward[1][i] = zeBackward[i][0];
//         zInitBackward[2][i] = zfBackward[i][0];
//     }

//     float** zmaxBlocksMaxBackward = malloc_matrix<float>(blocks+1, queryLen + 1);
//     // memcpy(zmaxBlocksMaxBackward[0], zmaxBackward, (queryLen + 1) * sizeof(float));

//     for (int b = 0; b < blocks; ++b) {
//         int start = b * length;
//         int end = (b + 1) * length;
//         forwardBackwardSaveBlockMaxLocal(scoreBackward, zInitBackward, vj, wj, T, go, ge, queryLen, start, end,
//                                          zmBackward, zeBackward, zfBackward,
//                                          zmaxBackward);
//         memcpy(zmaxBlocksMaxBackward[b+1], zmaxBackward, (queryLen + 1) * sizeof(float));
//     }

//     ///////////////////////////////////Rescale////////////////////////////////////////
//     // Rescale the values by the maximum in the log space for each block
//     // This turns the matrix into log space
//     rescaleBlocks(zmForward, zmaxBlocksMaxForward, queryLen, targetLen, length, blocks);
//     rescaleBlocks(zmBackward, zmaxBlocksMaxBackward, queryLen, targetLen, length, blocks);

//     ////////////////////////////////Check Output
//     // std::cout << "Forward matrix" << std::endl;
//     // for (size_t i=0; i < 8; ++i){
//     //     for (size_t j = 0; j < 8; ++j) {
//     //         std::cout << std::fixed << std::setprecision(5) << zmForward[i][j] << " ";
//     //     }
//     //     std::cout << std::endl;
//     // }
//     // std::cout << std::endl;

//     // std::cout << "Backward matrix" << std::endl;
//     // for (size_t i=0; i < 8; ++i){
//     //     for (size_t j=0; j < 8; ++j) {
//     //         std::cout << std::fixed << std::setprecision(5) << zmBackward[i][j] << " ";
//     //     }
//     //     std::cout << std::endl;
//     // }
//     // std::cout << std::endl;

//    // compute zm max
//     float max_zm = -DBL_MAX;
//     for (size_t i = 0; i < queryLen; ++i) {
//         for (size_t j = 0; j < targetLen; ++j) {
//             max_zm = std::max(max_zm, zmForward[i][j]);
//         }
//     }
//     // std::cout << "max zm\t" << max_zm << "\n";
//     // compute sum_exp
//     float sum_exp= 0.0;
//     for (size_t i = 0; i < queryLen; ++i) {
//         for (size_t j = 0; j < targetLen; ++j) {
//             sum_exp += exp(zmForward[i][j] - max_zm);
//         }
//     }
//     float logsumexp_zm = max_zm + log(sum_exp);
    
//     // compute posterior probabilities
//     float max_p = 0.0;
//     size_t max_i;
//     size_t max_j;
//     for (size_t i = 0; i < queryLen; ++i) {
//         for (size_t j = 0; j < targetLen; ++j) {
//             P[i][j] = exp(
//                 zmForward[i][j]
//                 + zmBackward[queryLen - 1 - i][targetLen - 1 - j]
//                 - log(scoreForward[i][j]) // FIXME scoreForward is already exp(S/T)
//                 - logsumexp_zm
//             );
//             if (P[i][j] > max_p) {
//                 max_p = P[i][j];
//                 max_i = i;
//                 max_j = j;
//             }
//         }
//     }

//     // traceback 
//     s_align result;
//     result.cigar = "";
//     result.cigar.reserve(queryLen + targetLen);
//     result.qEndPos1 = max_i;
//     result.dbEndPos1 = max_j;
//     float d;
//     float l;
//     float u;
//     size_t index = 0;
//     while (max_i > 0 && max_j > 0) {
//         d = P[max_i - 1][max_j - 1];
//         l = P[max_i][max_j - 1];
//         u = P[max_i - 1][max_j];
//         // std::cout << std::fixed << std::setprecision(8) << max_i << '\t' << max_j << '\t' << d << '\t' << l << '\t' << u << '\n';
//         if (d > l && d > u) {
//             max_i--;
//             max_j--;
//             result.cigar.push_back('M');
//         } else if (l > d && l > u) {
//             max_j--;
//             result.cigar.push_back('I');
//         } else {
//             max_i--;
//             result.cigar.push_back('D');
//         }
//     }
//     result.qStartPos1 = max_i;
//     result.dbStartPos1 = max_j;
//     result.cigarLen = result.cigar.length();
//     std::reverse(result.cigar.begin(), result.cigar.end());

//     free(ss1_num);
//     free(ss2_num);
//     free(aa1_num);
//     free(aa2_num);
//     delete[] zInitForward[0];
//     delete[] zInitForward[1];
//     delete[] zInitForward[2];
//     delete[] zInitBackward[0];
//     delete[] zInitBackward[1];
//     delete[] zInitBackward[2];

//     return result;
// }

// void FwBwAligner::forwardBackwardSaveBlockMaxLocal(float** S, float** z_init,
//                                                    float* vj, float* wj,
//                                                    float T, float go, float ge,
//                                                    size_t rows, size_t start, size_t end,
//                                                    float** zm, float** ze, float** zf, float* zmax) {
//     float exp_go = exp(go / T);
//     float exp_ge = exp(ge / T);
    
//     float** zmBlock = malloc_matrix<float>(rows + 1, end - start + 1);
//     float** zeBlock = malloc_matrix<float>(rows + 1, end - start + 1);
//     float** zfBlock = malloc_matrix<float>(rows + 1, end - start + 1);
    
//     //Init blocks
//     memset(zmBlock[0], 0, (end - start + 1) * sizeof(float));
//     memset(zeBlock[0], 0, (end - start + 1) * sizeof(float));
//     memset(zfBlock[0], 0, (end - start + 1) * sizeof(float));


//     // Initialize the first column of the segment starting from the second row
//     for (size_t i = 0; i < rows; ++i) {
//         zmBlock[i+1][0] = z_init[0][i];
//         zeBlock[i+1][0] = z_init[1][i];
//         zfBlock[i+1][0] = z_init[2][i];
//     }

//     size_t cols = end - start;
//     float* exp_ge_arr = static_cast<float*>(mem_align(16, cols * sizeof(float)));
//     for (size_t i = 0; i < cols; ++i) {
//         exp_ge_arr[i] = exp((i * ge + ge) / T);
//     }

//     float current_max = 0;
//     for (size_t i = 1; i <= rows; ++i) {
//         if (i != 1) {
//             zmBlock[i - 1][0] = exp(zmBlock[i - 1][0]);
//             zeBlock[i - 1][0] = exp(zeBlock[i - 1][0]);
//             zfBlock[i - 1][0] = exp(zfBlock[i - 1][0]);
//         }
//         for (size_t j = 1; j <= cols; ++j) {
//             float tmp = (zmBlock[i - 1][j - 1] + zeBlock[i - 1][j - 1] + zfBlock[i - 1][j - 1] + exp(-current_max));
//             zmBlock[i][j] = tmp * S[i - 1][start + j - 1];
//         }
        
//         float* zm_exp = static_cast<float*>(mem_align(16, cols * sizeof(float)));
//         memcpy(zm_exp, zmBlock[i], cols * sizeof(float));
//         zm_exp[0] = exp(zm_exp[0]);

//         // Correct translation of the cumulative sum
//         float cumulative_sum = 0;
//         for (size_t j = 1; j <= cols; ++j) {
//             cumulative_sum += zm_exp[j - 1] * vj[j - 1];
//             zeBlock[i][j] = (cumulative_sum / wj[j - 1]) + exp(zeBlock[i][0]) * exp_ge_arr[j - 1];
//         }
//         for (size_t j = 1; j <= cols; ++j) {
//             zfBlock[i][j] = (zmBlock[i - 1][j] * exp_go + zfBlock[i - 1][j] * exp_ge);
//         }


//         float z_temp = *std::max_element(zmBlock[i] + 1, zmBlock[i] + cols + 1);
//         zmax[i] = log(z_temp);
//         current_max += zmax[i];
//         for (size_t j = 1; j <= cols; ++j) {
//             zmBlock[i][j] /= z_temp;
//             zeBlock[i][j] /= z_temp;
//             zfBlock[i][j] /= z_temp;
//         }

//         for (size_t j = i; j <= rows; ++j) {
//             zmBlock[j][0] -= zmax[i];
//             zeBlock[j][0] -= zmax[i];
//             zfBlock[j][0] -= zmax[i];
//         }

//         free(zm_exp);
//     }

//     //Calculate the cumulative sum of zmax[1:]
//     std::vector<float> rescale(rows);
//     // std::partial_sum(zmax + 1, zmax + rows + 1, rescale.begin());
//     std::partial_sum(zmax + 1, zmax + rows + 1, rescale.begin());

//     for (size_t i = 0; i < rows; ++i) {
//         z_init[0][i] = log(zmBlock[i + 1][cols]) + rescale[i];
//         z_init[1][i] = log(zeBlock[i + 1][cols]) + rescale[i];
//         z_init[2][i] = log(zfBlock[i + 1][cols]) + rescale[i];
//     }

//     for (size_t i = 0; i < rows; ++i) {
//         memcpy(zm[i] + start, zmBlock[i+1]+1, (end - start) * sizeof(float));
//         memcpy(ze[i] + start, zeBlock[i+1]+1, (end - start) * sizeof(float));
//         memcpy(zf[i] + start, zfBlock[i+1]+1, (end - start) * sizeof(float));
//     }

//     free(zmBlock);
//     free(zeBlock);
//     free(zfBlock);
//     free(exp_ge_arr);
// }

// void FwBwAligner::rescaleBlocks(float **matrix, float **scale, size_t rows, size_t cols, int length, int blocks){
//     // Function to rescale the values by the maximum in the log space for each block
//     for (size_t b = 0; b < blocks; ++b) {
//         size_t start = b * length;
//         size_t end = (b + 1) * length;
//         std::vector<float> cumsum(rows);
//         std::partial_sum(scale[b + 1] + 1, scale[b + 1] + 1 + rows, cumsum.begin());
//         for (size_t i = 0; i < rows; ++i) {
//             for (size_t j = start; j < end; ++j) {
//                 // matrix[i][j] = log(matrix[i][j]) ;// + cumsum[i-1];
//                 matrix[i][j] = log(matrix[i][j]) + cumsum[i];
//             }
//         }
//     }
// }

int fwbw(int argc, const char **argv, const Command &command) {
    Parameters &par = Parameters::getInstance();
    par.parseParameters(argc, argv, command, true, 0, MMseqsParameter::COMMAND_ALIGN);
    // FwBwAligner fwbwAln(par.db1, par.db2, par.db3, par.db4, par.db4Index);

    
}
