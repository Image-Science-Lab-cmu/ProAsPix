function meas = load_processed_data(fname, folder1, folder2)

%%%LOADING DATASET
meas = load([folder1 '/' fname '/data.mat']);
meas3 = load([folder2 '/' fname '.mat']);

meas3.assort_restored = permute(meas3.assort_restored,[2 3 1]);

meas.assort_restored = 0*meas.assort_sim;
meas.assort_restored(:, 108+(1:1024), :) = meas3.assort_restored;

%%Crop to align with restored iamgery
meas.assort_index = meas.assort_index(:, 108+(1:1024), :);
meas.assort_meas = meas.assort_meas(:, 108+(1:1024), :);
meas.assort_sim = meas.assort_sim(:, 108+(1:1024), :);
meas.assort_restored = meas.assort_restored(:, 108+(1:1024), :);
meas.guide = meas.guide(:, 108+(1:1024), :);
meas.full_scan = meas.full_scan(:, 108+(1:1024), :);

end