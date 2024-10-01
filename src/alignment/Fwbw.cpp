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

int fwbw(int argc, const char **argv, const Command &command) {
    Parameters &par = Parameters::getInstance();
    par.parseParameters(argc, argv, command, true, 0, 0);

    // Load alnDB
    DBReader<unsigned int> tdbr(par.db3.c_str(), par.db3Index.c_str(), par.threads, DBReader<unsigned int>::USE_INDEX|DBReader<unsigned int>::USE_DATA);
    tdbr.open(DBReader<unsigned int>::NOSORT);

    // Load scoring matrix
    const int targetSeqType = tdbr.getDbtype();
    int gapOpen, gapExtend;
    BaseMatrix *subMat;

    if (Parameters::isEqualDbtype(targetSeqType, Parameters::DBTYPE_NUCLEOTIDES)) {
        subMat = new NucleotideMatrix(par.scoringMatrixFile.values.nucleotide().c_str(), 1.0, par.scoreBias);
        gapOpen = par.gapOpen.values.nucleotide();
        gapExtend = par.gapExtend.values.nucleotide();
    } else {
        subMat = new SubstitutionMatrix(par.scoringMatrixFile.values.aminoacid().c_str(), 2.0, par.scoreBias);
        gapOpen = par.gapOpen.values.aminoacid();
        gapExtend = par.gapExtend.values.aminoacid();
    }


    // Load databases
    qDbrIdx = new IndexReader(par.db1, par.threads,
                                extended & Parameters::DBTYPE_EXTENDED_INDEX_NEED_SRC ? IndexReader::SRC_SEQUENCES : IndexReader::SEQUENCES,
                                (touch) ? IndexReader::PRELOAD_INDEX : 0);
    qdbr = qDbrIdx->sequenceReader;
    querySeqType = qdbr->getDbtype();
    tDbrIdx = new IndexReader(par.db2, par.threads,
                                extended & Parameters::DBTYPE_EXTENDED_INDEX_NEED_SRC ? IndexReader::SRC_SEQUENCES : IndexReader::SEQUENCES,
                                (touch) ? IndexReader::PRELOAD_INDEX : 0);
    tdbr = tDbrIdx->sequenceReader;
    targetSeqType = tdbr->getDbtype();
    DBReader<unsigned int> resultReader(par.db3.c_str(), par.db3Index.c_str(), par.threads, DBReader<unsigned int>::USE_INDEX | DBReader<unsigned int>::USE_DATA);
    resultReader->open(DBReader<unsigned int>::LINEAR_ACCCESS);
    // Open DBWriter
    DBWriter writer(par.db4.c_str(), par.db4Index.c_str(), par.threads, par.compressed, Parameters::DBTYPE_ALIGNMENT_RES);


    // Iteration size
    const size_t flushSize = 100000000; // buffer size for batch processing
    size_t iterations = static_cast<int>(ceil(static_cast<double>(resultReader.getSize()) / static_cast<double>(flushSize)));

    for (size_t i = 0; i < iterations; ++i) {
        size_t start = (i * flushSize);
        size_t bucketSize = std::min(resultReader.getSize() - (i * flushSize), flushSize);
        DEBUG::Progress progress(bucketSize);
        //Question: what is purpose of progress and result reserve?
    

    // OpenMP
#pragma omp parallel
    {
        unsigned int thread_idx = 0;
#ifdef OPENMP
        thread_idx = (unsigned int)omp_get_thread_num();
#endif
        char buffer[1024 + 32768*4];
        Sequence query(par.maxSeqLen, targetSeqType, subMat, 0, false, par.compBiasCorrection);
        Sequence target(par.maxSeqLen, targetSeqType, subMat, 0, false, par.compBiasCorrection);
#pragma omp for schedule(dynamic, 1)
        for (size_t id = start; id < (start + bucketSize); ++id) {
            progress.updateProgress();
            // Parse alnDB
            char *data = resultReader.getData(id, thread_idx);
            unsigned int queryDbKey = resultReader->getDbKey(id);
            

            while (*results != '\0') {
                Util::parseKey(results, dbKey);
                const unsigned int key = (unsigned int)strtoul(dbKey, NULL, 10);
            }
            Util::parseKey(results, dbKey);
            const unsigned int key = (unsigned int) strtoul(dbKey, NULL, 10);
            const size_t edgeId = seqReader.getId(key);
            resultWriter.writeData(seqReader.getData(edgeId, thread_idx), seqReader.getEntryLen(edgeId) - 1, resultReader.getDbKey(id), thread_idx);



            while (*data != '\0') {
                Util::parseKey(data, buffer);
                const unsigned int key = (unsigned int) strtoul(buffer, NULL, 10);
                results.push_back(key);
                data = Util::skipLine(data);
            }



            runFwbwAlgorithm(querySeq, targetSeq, resultBuffer);

            // Write
            writer.writeData(fwbwAlnResultsOutString.c_str(), fwbwAlnResultsOutString.length(), queryDbKey, thread_idx);

            // Cleanup
            fwbwAlnResultsOutString.clear();

            //FIXME
            writer.writeStart(thread_idx);
            writer.writeAdd(resultBuffer, strlen(resultBuffer), thread_idx);
            writer.writeEnd(key, thread_idx);
        }
    }
    tbdr.remapData();
    }
    writer.close();
    tdbr.close();
    delete subMat;

    return EXIT_SUCCESS;
}


