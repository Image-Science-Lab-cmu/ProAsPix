function hsi_est = reconstruct_csmusi(full_scan, hsi_spec, hsi_wvl, num)


meas_idx = randperm(256,num);

%%Part 2 --- measurements
y0 = double(full_scan(:,:,meas_idx));
y0 = (1/sqrt(5))*poissrnd(y0*sqrt(5));
y0 = y0/2^16;

y0 = squeeze(y0);
siz = [size(y0, 1), size(y0, 2)];

nSLM = size(y0, 3);
y0 = reshape(y0, [], num)';
nSLM = size(y0, 1);

A3 = hsi_spec(meas_idx, :);
A3norm = norm(A3);
A3 = A3/norm(A3);
A3inv = (A3'*A3 + 0.002*eye(size(A3, 2)))\A3';


%%%PArt 3 ---- RECONSTRUCT
hsi_est = A3inv*y0;
%rec_hsi = rec_hsi - ones(size(rec_hsi, 1),1)*min(rec_hsi, [], 1);
hsi_est = reshape(hsi_est', siz(1), siz(2), length(hsi_wvl));
hsi_est = max(0, hsi_est);
hsi_est = hsi_est/norm(hsi_est(:));


end

