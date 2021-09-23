function rec_hsi = reconstruct_full_scan(y0, A2, A2_lambda)

A2norm = norm(A2);
A2 = A2/norm(A2);
A2inv = (A2'*A2 + 0.002*eye(size(A2, 2)))\A2';


%%Part 2 --- measurements
y0 = double(y0)/2^16;
y0 = squeeze(y0);
siz = [size(y0, 1), size(y0, 2)];

nSLM = size(y0, 3);
y0 = reshape(y0, [], nSLM)';
nSLM = size(y0, 1);


%%%PArt 3 ---- RECONSTRUCT
rec_hsi = A2inv*y0;
%rec_hsi = rec_hsi - ones(size(rec_hsi, 1),1)*min(rec_hsi, [], 1);
rec_hsi = reshape(rec_hsi', siz(1), siz(2), length(A2_lambda));


end