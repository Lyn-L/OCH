function evaluation_info = evaluateOCH(data, param)


groundtruth = data.groundtruth;
traindata = data.train_data';
dbdata = data.db_data';
tstdata = data.test_data';

timerTrain = tic;
[ R, eigVec, sample_mean] = trainOCH(traindata, param);
trainT=toc(timerTrain);

B_db = compressPOGH(dbdata, R, eigVec, sample_mean);
if(isfield(data, 'groundtruth'))

    timeTest = tic;
    B_tst = compressOCH(tstdata, R, eigVec, sample_mean);
    compressT=toc(timeTest)/size(B_tst,1);

    evaluation_info = performance(B_tst, B_db, groundtruth, param);
    evaluation_info.trainT = trainT;
    evaluation_info.compressT = compressT;
else
    D_dist =  -hammingDist(B_db,B_db);
    evaluation_info.AP = compute_avg_top(D_dist);
end


end
