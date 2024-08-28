#include "Util.h"
#include "Parameters.h"
#include "Matcher.h"
#include "Debug.h"
#include "DBReader.h"
#include "DBWriter.h"
#include "NucleotideMatrix.h"
#include "SubstitutionMatrix.h"
#include "Alignment.h"
#include "itoa.h"
#include "Timer.h"

#ifdef OPENMP
#include <omp.h>
#endif

struct __attribute__((__packed__)) ProteomeEntry{
    int proteomeKey;
    unsigned int proteomeAALen;
    int repProtKey;
    float protSimScore;
    unsigned int memCount;
    float totalSeqId;

    ProteomeEntry(unsigned int proteomeKey = -1, unsigned int pLength = 0, int repKey = -1, float simScore = 0.0f, unsigned int memCount = 0, float seqId = 0.0f)
        : proteomeKey(proteomeKey), proteomeAALen(pLength), repProtKey(repKey), protSimScore(simScore), memCount(memCount), totalSeqId(seqId) {}

    int getRepProtKey() {
        return repProtKey;
    }
    int getProteomeKey() {
        return proteomeKey;
    }
    bool isCovered(){
        if (repProtKey == -1) {
            return false;
        }
        return true;
    }
    void computeRedundancy() {
        protSimScore = totalSeqId / proteomeAALen;

    }
    void addSeqId(float seqId) {
        totalSeqId += seqId;
    }
    void addSeqLen(unsigned int seqLength) {
        proteomeAALen += seqLength;
    }
    void resetProteomeInfo(){
        memCount = 0;
        totalSeqId = 0.0f;
    }
    float getProtSimScore() {
        return protSimScore;
    }
    static bool compareByKey(const ProteomeEntry& a, const ProteomeEntry& b) {
        
        if (a.repProtKey < b.repProtKey){
            return true;
        }
        if (a.repProtKey > b.repProtKey){
            return false;
        }
        if (a.proteomeKey == a.repProtKey && b.proteomeKey != b.repProtKey){
            return true;
        }
        if (a.proteomeKey != a.repProtKey && b.proteomeKey == b.repProtKey){
            return false;
        }
        return false;
    }
};

struct __attribute__((__packed__)) MemberProteinEntry{
    unsigned int proteomeKey;
    unsigned int proteinKey;
};

struct ClusterEntry {
    bool isAvailable;
    std::vector<MemberProteinEntry> memberProteins;

    ClusterEntry() : isAvailable(false) {}

    ClusterEntry(size_t memberProteinSize) : isAvailable(true) {
        memberProteins.reserve(memberProteinSize);
    }

    void resetClusterInfo(){
        memberProteins.clear();
        isAvailable = false;
    }
};

void calculateProteomeLength(std::vector<ProteomeEntry>& proteomeList, DBReader<unsigned int>::LookupEntry* const& lookup, size_t lookupSize, DBReader<unsigned int>& tProteinDB) {
    for (size_t i = 0; i < lookupSize; i++) {
        const unsigned int ProteomeId = lookup[i].fileNumber;
        const unsigned int ProteinId = lookup[i].id;
        proteomeList[ProteomeId].addSeqLen(tProteinDB.getSeqLen(ProteinId));
        if (proteomeList[ProteomeId].proteomeKey == -1) {
            proteomeList[ProteomeId].proteomeKey = ProteomeId;
        }
    }
}

void initLocalClusterReps(size_t& id, std::vector<ClusterEntry>& localClusterReps, std::vector<unsigned int>& localMemCount, DBReader<unsigned int>& tProteinDB, DBReader<unsigned int>& linResDB, unsigned int thread_idx){
    std::vector<unsigned int> memberKeys;
    memberKeys.reserve(50); // store key for every protein in a cluster

    std::vector<uint8_t> isProteomeInCluster(localMemCount.size(), 0);
    char buffer[1024 + 32768*4];
    char *clustData = linResDB.getData(id, thread_idx);
    
    while (*clustData != '\0') {
        Util::parseKey(clustData, buffer);
        const unsigned int key = (unsigned int) strtoul(buffer, NULL, 10);
        memberKeys.push_back(key);
        clustData = Util::skipLine(clustData);
    }
    if (memberKeys.size() > 1) { //If not a singleton cluster
        ClusterEntry eachClusterRep(memberKeys.size());
        //init MemberProteinEntry and add it to memberProteins vector
        for (auto& eachMemberKey : memberKeys) {
            const unsigned int proteinId = tProteinDB.getId(eachMemberKey);
            const unsigned int proteomeKey = tProteinDB.getLookupFileNumber(proteinId);
            MemberProteinEntry mem;
            mem.proteomeKey = proteomeKey;
            mem.proteinKey = proteinId;
            eachClusterRep.memberProteins.push_back(mem);
            if (isProteomeInCluster[proteomeKey] == 0) {
                isProteomeInCluster[proteomeKey] = 1;
            }
        }
        localClusterReps.push_back(eachClusterRep);
    }

    for (size_t i = 0; i < localMemCount.size(); i++) {
        if (isProteomeInCluster[i]) {
            localMemCount[i]++;
        }
    }
}

bool FindNextRepProteome(std::vector<ProteomeEntry>& proteomeList, unsigned int& RepProteomeId) {
    bool isAllCovered = true;
    
    unsigned int maxMemberCount = 0;
    unsigned int notCoveredProtCount = 0;

    for (size_t idx = 0; idx < proteomeList.size(); idx++) {
        if (proteomeList[idx].isCovered()) {
            continue;
        }else{
            isAllCovered = false;
            notCoveredProtCount++;
            if (proteomeList[idx].memCount > maxMemberCount) {
                maxMemberCount = proteomeList[idx].memCount;
                RepProteomeId = idx;
            }
        }
    }

    if (isAllCovered){
        return false;
    }else if (notCoveredProtCount == 1) {
        //last one and it is singleton
        proteomeList[RepProteomeId].repProtKey = RepProteomeId;
        proteomeList[RepProteomeId].protSimScore = 1.0;
        return false;
    }else{
        return true;
    }
}

void runAlignmentForCluster(const ClusterEntry& clusterRep, unsigned int RepProteomeId, DBReader<unsigned int>& tProteinDB, Matcher& matcher, Sequence& query, Sequence& target, std::vector<ProteomeEntry>& proteomeList, Parameters& par, unsigned int thread_idx, int swMode, std::vector<float>& localSeqIds, DBWriter& proteinClustWriter) {
    char buffer[1024]; 
    bool isRepFound = false;
    unsigned int lastqLen = 0;
    unsigned int qproteinKey = 0;
    unsigned int qproteomeKey =0;
    //find Rep
    for (auto& eachMember : clusterRep.memberProteins){
        if (eachMember.proteomeKey == RepProteomeId) {
            isRepFound = true;
            const unsigned int queryId = eachMember.proteinKey;
            if( lastqLen < tProteinDB.getSeqLen(queryId)){
                lastqLen = tProteinDB.getSeqLen(queryId);
                qproteinKey = eachMember.proteinKey;
                qproteomeKey = eachMember.proteomeKey;
            }
        }
    }
    if (isRepFound){
        proteinClustWriter.writeStart(thread_idx);
        const unsigned int queryId = qproteinKey;
        const unsigned int queryKey = tProteinDB.getDbKey(queryId);
        char* querySeq = tProteinDB.getData(queryId, thread_idx); 
        query.mapSequence(queryId, queryKey, querySeq, tProteinDB.getSeqLen(queryId));
        matcher.initQuery(&query);
        const unsigned int queryProteomeLength = proteomeList[qproteomeKey].proteomeAALen;

        // representative protein :same query and target (for createtsv)
        Matcher::result_t rep_reult = matcher.getSWResult(&query, INT_MAX, false, par.covMode, par.covThr, par.evalThr, swMode, par.seqIdMode, true);
        size_t rep_len = Matcher::resultToBuffer(buffer, rep_reult, par.addBacktrace);
        proteinClustWriter.writeAdd(buffer, rep_len, thread_idx);
        localSeqIds[qproteomeKey] += rep_reult.getSeqId()*query.L;

        for (auto& eachTargetMember : clusterRep.memberProteins){
            if (eachTargetMember.proteomeKey == RepProteomeId) {
                continue;
            }
            // if query Proteome's length < target Proteome's length * 0.9, skip
            const unsigned int targetProteomeLength = proteomeList[eachTargetMember.proteomeKey].proteomeAALen;
            if (queryProteomeLength < targetProteomeLength * 0.9) {
                continue;
            }
            const unsigned int targetId = eachTargetMember.proteinKey;
            const unsigned int targetKey = tProteinDB.getDbKey(targetId);
            unsigned int tproteomeKey = eachTargetMember.proteomeKey;

            char* targetSeq = tProteinDB.getData(targetId, thread_idx);
            target.mapSequence(targetId, targetKey, targetSeq, tProteinDB.getSeqLen(targetId));

            if (Util::canBeCovered(par.covThr, par.covMode, query.L, target.L) == false) {
                continue;
            }

            const bool isIdentity = (queryId == targetId && par.includeIdentity) ? true : false;
            Matcher::result_t result = matcher.getSWResult(&target, INT_MAX, false, par.covMode, par.covThr, par.evalThr, swMode, par.seqIdMode, isIdentity);

            if (Alignment::checkCriteria(result, isIdentity, par.evalThr, par.seqIdThr, par.alnLenThr, par.covMode, par.covThr)) {
                if (query.L >= target.L*0.9){
                    size_t len = Matcher::resultToBuffer(buffer, result, par.addBacktrace);
                    proteinClustWriter.writeAdd(buffer, len, thread_idx);
                    localSeqIds[tproteomeKey] += result.getSeqId()*target.L;
                }
            }
        }
        proteinClustWriter.writeEnd(queryKey, thread_idx);
    }
}

bool updateproteomeList(std::vector<ProteomeEntry>& proteomeList, const unsigned int& RepProteomeId){
    bool isRepSingleton = true;

    for (size_t i = 0; i < proteomeList.size(); i++) {
        if (proteomeList[i].isCovered() == false) {
            if (i == RepProteomeId){
                proteomeList[i].repProtKey = RepProteomeId;
                proteomeList[i].protSimScore = 1.0;
            }else{
                proteomeList[i].computeRedundancy();
                if (proteomeList[i].getProtSimScore() >= 0.9) {
                    proteomeList[i].repProtKey = RepProteomeId;
                    isRepSingleton = false;
                }else{
                    proteomeList[i].resetProteomeInfo();
                }
            }
        }
    }
    return isRepSingleton;
}

void updateClusterReps(ClusterEntry& clusterRep, std::vector<ProteomeEntry>& proteomeList, std::vector<unsigned int>& localMemCount){
    std::vector<uint8_t> isProteomeInCluster(localMemCount.size(), 0);
    if (clusterRep.isAvailable) {
        bool isAllMemberCovered = true;
        unsigned int notCoveredCount = 0;
        unsigned int lastProteomeKey = 0;
        //update cluster member info
        for (auto& eachMember : clusterRep.memberProteins) {
            if (proteomeList[eachMember.proteomeKey].isCovered() == false) {
                notCoveredCount++;
                lastProteomeKey = eachMember.proteomeKey;
                isAllMemberCovered = false;
                if (isProteomeInCluster[eachMember.proteomeKey] == 0) {
                    isProteomeInCluster[eachMember.proteomeKey] = 1;
                }
            }
        }
        if (isAllMemberCovered) {
            clusterRep.resetClusterInfo(); //set clusterRep.isAvailable = false;
        }
        if (notCoveredCount == 1) { //singleton Cluster
            isProteomeInCluster[lastProteomeKey] = 0;
            clusterRep.resetClusterInfo(); //set clusterRep.isAvailable = false;
        }
    }

    //update localMemCount
    if (clusterRep.isAvailable) {
        for (size_t i=0; i < localMemCount.size(); i++) {
            if (isProteomeInCluster[i]) {
                localMemCount[i]++;
            }
        }
    }

}


void writeProteomeClusters(DBWriter &proteomeClustWriter, std::vector<ProteomeEntry> &proteomeList) {
    std::vector<size_t> proteomeClusterIdx;
    char proteomeBuffer[1024];
    int repProtIdCluster = proteomeList[0].getRepProtKey();

    for (size_t idx = 0; idx < proteomeList.size(); idx++) {
        int repProtId = proteomeList[idx].getRepProtKey();
        if (repProtIdCluster != repProtId) {
            proteomeClustWriter.writeStart();
            for (auto &eachIdx : proteomeClusterIdx) {
                char *basePos = proteomeBuffer;
                char *tmpProteomeBuffer = Itoa::i32toa_sse2(proteomeList[eachIdx].getProteomeKey(), proteomeBuffer);
                *(tmpProteomeBuffer - 1) = '\t';
                tmpProteomeBuffer = Util::fastSeqIdToBuffer(proteomeList[eachIdx].getProtSimScore(), tmpProteomeBuffer);
                *(tmpProteomeBuffer - 1) = '\n';
                proteomeClustWriter.writeAdd(proteomeBuffer, tmpProteomeBuffer - basePos);
            }
            proteomeClustWriter.writeEnd(repProtIdCluster);
            // Reset
            repProtIdCluster = repProtId;
            proteomeClusterIdx.clear();
            proteomeClusterIdx.push_back(idx);
        } else {
            proteomeClusterIdx.push_back(idx);
        }

        if (idx == proteomeList.size() - 1) {
            proteomeClustWriter.writeStart();
            for (auto &eachIdx : proteomeClusterIdx) {
                char *basePos = proteomeBuffer;
                char *tmpProteomeBuffer = Itoa::i32toa_sse2(proteomeList[eachIdx].getProteomeKey(), proteomeBuffer);
                *(tmpProteomeBuffer - 1) = '\t';
                tmpProteomeBuffer = Util::fastSeqIdToBuffer(proteomeList[eachIdx].getProtSimScore(), tmpProteomeBuffer);
                *(tmpProteomeBuffer - 1) = '\n';
                proteomeClustWriter.writeAdd(proteomeBuffer, tmpProteomeBuffer - basePos);
            }
            proteomeClustWriter.writeEnd(repProtIdCluster);
            proteomeClusterIdx.clear();
        }
    }
}


int proteomecluster(int argc, const char **argv, const Command &command){
    //Initialize parameters
    Parameters &par = Parameters::getInstance();
    par.overrideParameterDescription(par.PARAM_ALIGNMENT_MODE, "How to compute the alignment:\n0: automatic\n1: only score and end_pos\n2: also start_pos and cov\n3: also seq.id", NULL, 0);
    par.parseParameters(argc, argv, command, true, 0, 0);

    if (par.alignmentMode == Parameters::ALIGNMENT_MODE_UNGAPPED) {
        Debug(Debug::ERROR) << "Use rescorediagonal for ungapped alignment mode.\n";
        EXIT(EXIT_FAILURE);
    }
    if (par.addBacktrace == true) {
        par.alignmentMode = Parameters::ALIGNMENT_MODE_SCORE_COV_SEQID;
    }
    
    unsigned int swMode = Alignment::initSWMode(par.alignmentMode, par.covThr, par.seqIdThr);

    //Open the target protein database
    DBReader<unsigned int> tProteinDB(par.db1.c_str(), par.db1Index.c_str(), par.threads, DBReader<unsigned int>::USE_DATA|DBReader<unsigned int>::USE_INDEX|DBReader<unsigned int>::USE_LOOKUP|DBReader<unsigned int>::USE_SOURCE);
    tProteinDB.open(DBReader<unsigned int>::NOSORT);
    const int tProteinSeqType = tProteinDB.getDbtype();

    if (par.preloadMode != Parameters::PRELOAD_MODE_MMAP) {
        tProteinDB.readMmapedDataInMemory();
    }
    //Open lookup table to get the source of the protein and the protein length
    DBReader<unsigned int>::LookupEntry* tLookup = tProteinDB.getLookup();
    const size_t tLookupSize = tProteinDB.getLookupSize();
    unsigned int totalProteomeNumber = tLookup[tLookupSize - 1].fileNumber;
    std::vector<ProteomeEntry> proteomeList(totalProteomeNumber + 1);
    calculateProteomeLength(proteomeList, tLookup, tLookupSize, tProteinDB);

    //Open the linclust result
    DBReader<unsigned int> linResDB(par.db2.c_str(), par.db2Index.c_str(), par.threads, DBReader<unsigned int>::USE_DATA|DBReader<unsigned int>::USE_INDEX);
    linResDB.open(DBReader<unsigned int>::LINEAR_ACCCESS);

    //Open the DBWriter
    DBWriter proteinClustWriter(par.db3.c_str(), par.db3Index.c_str(), par.threads, par.compressed, Parameters::DBTYPE_GENERIC_DB);
    proteinClustWriter.open();
    int proteomeDBType = DBReader<unsigned int>::setExtendedDbtype(Parameters::DBTYPE_GENERIC_DB, Parameters::DBTYPE_EXTENDED_SRC_IDENTIFIER);
    DBWriter proteomeClustWriter(par.db4.c_str(), par.db4Index.c_str(), 1, par.compressed, proteomeDBType);
    proteomeClustWriter.open();

    std::vector<ClusterEntry> clusterReps; 

    int gapOpen, gapExtend;
    // BaseMatrix *subMat;
    SubstitutionMatrix subMat(par.scoringMatrixFile.values.aminoacid().c_str(), 2.0, par.scoreBias);
    gapOpen = par.gapOpen.values.aminoacid();
    gapExtend = par.gapExtend.values.aminoacid();
    EvalueComputation evaluer(tProteinDB.getAminoAcidDBSize(), &subMat, gapOpen, gapExtend);
    Debug(Debug::INFO) << "Initilize ";
    Timer timer;
    // Debug(Debug::INFO) << "Start Initialization\n";
    #pragma omp parallel
    {
        unsigned int thread_idx = 0;
    #ifdef OPENMP
        thread_idx = (unsigned int) omp_get_thread_num();
    #endif    
        std::vector<ClusterEntry> localClusterReps;
        std::vector<unsigned int> localMemCount(proteomeList.size(), 0);

        #pragma omp for schedule(dynamic, 1)
        for (size_t id = 0; id < linResDB.getSize(); id++) {
            initLocalClusterReps(id, localClusterReps, localMemCount, tProteinDB, linResDB, thread_idx);
        }

        for (size_t idx = 0; idx < localMemCount.size(); idx++) {
        __sync_fetch_and_add(&proteomeList[idx].memCount, localMemCount[idx]);
        }

        #pragma omp critical
        {
            clusterReps.insert(clusterReps.end(),
                               std::make_move_iterator(localClusterReps.begin()),
                               std::make_move_iterator(localClusterReps.end()));
        }

    }
    Debug(Debug::INFO) << timer.lap() << "\n";

    unsigned int RepProteomeId = -1;
    // Debug(Debug::INFO) << "Run Alignment\n";
    Debug(Debug::INFO) << "Proteome Clustering ";
    timer.reset();
    while (FindNextRepProteome(proteomeList, RepProteomeId)) {
        #pragma omp parallel
        {
            unsigned int thread_idx = 0;
        #ifdef OPENMP
            thread_idx = (unsigned int) omp_get_thread_num();
        #endif  
            Matcher matcher(tProteinSeqType, tProteinSeqType, par.maxSeqLen, &subMat, &evaluer, par.compBiasCorrection, par.compBiasCorrectionScale, gapOpen, gapExtend, 0.0, par.zdrop);
            Sequence query(par.maxSeqLen, tProteinSeqType, &subMat, 0, false, par.compBiasCorrection);
            Sequence target(par.maxSeqLen, tProteinSeqType, &subMat, 0, false, par.compBiasCorrection);
            std::vector <float> localSeqIds(proteomeList.size(), 0.0f);
            #pragma omp for schedule(dynamic, 1)
            for (size_t i = 0; i < clusterReps.size(); i++) {
                if (clusterReps[i].isAvailable) {
                    runAlignmentForCluster(clusterReps[i], RepProteomeId, tProteinDB, matcher, query, target, proteomeList, par, thread_idx, swMode, localSeqIds, proteinClustWriter);
                }
                
            }
            #pragma omp critical
            {
                for (size_t idx = 0; idx < proteomeList.size(); idx++) {
                    proteomeList[idx].addSeqId(localSeqIds[idx]);
                }
            }
        }

        // Debug(Debug::INFO) << "2. Update ProteomeDB. Calculate the similarity score and check redundancy | Rep Proteome id : " << RepProteomeId << "\n";
        bool isRepSingleton = updateproteomeList(proteomeList, RepProteomeId);

        if (isRepSingleton) {
            proteomeList[RepProteomeId].repProtKey = RepProteomeId;
            proteomeList[RepProteomeId].protSimScore = 1.0;
        }

        // Debug(Debug::INFO) << "3. Re-Setup Proteome and ClusterReps\n";
        #pragma omp parallel
        {
            std::vector<unsigned int> localMemCount(proteomeList.size(), 0);
            #pragma omp for schedule(dynamic, 1)
            for (size_t i = 0; i < clusterReps.size(); i++) {
                updateClusterReps(clusterReps[i], proteomeList, localMemCount);
            }

            for (size_t i = 0; i < proteomeList.size(); i++) {
                __sync_fetch_and_add(&proteomeList[i].memCount, localMemCount[i]);
            }
        }
    }
    Debug(Debug::INFO) << timer.lap() << "\n";
    //sort proteomeList by repProtKey
    SORT_PARALLEL(proteomeList.begin(), proteomeList.end(), ProteomeEntry::compareByKey);

    //write result of proteome clustering
    writeProteomeClusters(proteomeClustWriter, proteomeList);

    proteomeClustWriter.close();
    proteinClustWriter.close();
    tProteinDB.close();
    linResDB.close();
    return EXIT_SUCCESS;
}