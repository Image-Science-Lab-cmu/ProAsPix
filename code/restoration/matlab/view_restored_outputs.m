% Save meas, target, restored, measdiff, and restoreddiff for all patterns
% and guide image, all as patches from the desired scene output mat file.
% 
% Author: Vijay Rengarajan

clc;
clear;
%%
close all;
%%
load('../saved_outputs/v1_without_guide/0625.1049.MatroshkaFamily.mat');
% load('../saved_outputs/v1_without_guide/0806.1542.Plants3.mat');

assort_gt = permute(assort_gt, [2, 3, 1]);
assort_meas = permute(assort_meas, [2, 3, 1]);
assort_restored = permute(assort_restored, [2, 3, 1]);
% guide_image = permute(guide_image, [2, 3, 1]);
%%
if 0
% To imshow patches at different locations for saving later.

pattern_idx = 5; % 1d horizontal
    
% pattern_idx = 9; % diag mirrored
% pattern_idx = 16; % horizontal staggered
% pattern_idx = 17; % horizontal staggered

% MatroshkaFamily
% slice_r = 200 + [1:128];
% slice_c = 300 + [1:128];

% slice_r = 0 + [201:350];
% slice_c = 0 + [201:350];

% slice_r = 0 + [1:1024];
% slice_c = 0 + [1:1024];

slice_r = 300 + [1:128];
slice_c = 300 + [1:128];


meas = assort_meas(slice_r, slice_c, pattern_idx);
target = assort_gt(slice_r, slice_c, pattern_idx);
restored = assort_restored(slice_r, slice_c, pattern_idx);
% guide = guide_image(slice_r, slice_c, :);



figure;
subplot(3,2,1), imshow(target, []);
% subplot(3,2,2), imshow(guide);
subplot(3,2,3), imshow(meas, []);
subplot(3,2,4), imshow(10*abs(meas-target), []);
subplot(3,2,5), imshow(restored, []);
subplot(3,2,6), imshow(10*abs(restored-target), []);
end
%%
if 0
% Actual saving patches occurs here.

save_image_path = '../saved_images/v1_without_guide_patches/';
% save_image_path = '../saved_images/v1_without_guide_patches_Plants3/';

guide_pending = 0;

for pattern_idx = 1:92 % For MatroshkaFamily
% for pattern_idx = 1:5 % For Plants3
    fprintf('Saving pattern_idx %d\n', pattern_idx);
    % For MatroshkaFamily (trained patterns)
    slice_r = 200 + [1:128];
    slice_c = 300 + [1:128];
    
    % For Plants3 (new pattern)
    % slice_r = 300 + [1:128];
    % slice_c = 300 + [1:128];

    meas = assort_meas(slice_r, slice_c, pattern_idx);
    target = assort_gt(slice_r, slice_c, pattern_idx);
    restored = assort_restored(slice_r, slice_c, pattern_idx);
    
    meas = meas * 65535.0; meas(meas > 65535.0) = 65535.0; meas(meas < 0) = 0;
    target = target * 65535.0; target(target > 65535.0) = 65535.0; target(target < 0) = 0;
    restored = restored * 65535.0; restored(restored > 65535.0) = 65535.0; restored(restored < 0) = 0;
    
    meas = uint16(meas);
    target = uint16(target);
    restored = uint16(restored);
    
    % For Plants3
    % meas = 2 * meas;
    % target = 2 * target;
    % restored = 2 * restored;
    
    if guide_pending == 1
        guide = guide_image(slice_r, slice_c, :);
        guide = guide * 255.0; guide(guide > 255.0) = 255.0; guide(guide < 0) = 0;
        guide = uint8(guide);
        imwrite(guide, [save_image_path, '/px_guide.png']);
        guide_pending = 0;
    end

    imwrite(target, [save_image_path, '/p', num2str(pattern_idx), '_target.png']);
    imwrite(meas, [save_image_path, '/p', num2str(pattern_idx), '_meas.png']);
    imwrite(10*abs(meas-target), [save_image_path, '/p', num2str(pattern_idx), '_measdiff.png']);
    imwrite(restored, [save_image_path, '/p', num2str(pattern_idx), '_restored.png']);
    imwrite(10*abs(restored-target), [save_image_path, '/p', num2str(pattern_idx), '_restoreddiff.png']);

    % For Plants3
%     imwrite(target, [save_image_path, '/p', num2str(pattern_idx), '_target.png']);
%     imwrite(meas, [save_image_path, '/p', num2str(pattern_idx), '_meas.png']);
%     imwrite(5*abs(meas-target), [save_image_path, '/p', num2str(pattern_idx), '_measdiff.png']);
%     imwrite(restored, [save_image_path, '/p', num2str(pattern_idx), '_restored.png']);
%     imwrite(5*abs(restored-target), [save_image_path, '/p', num2str(pattern_idx), '_restoreddiff.png']);

end

end

%%
if 1
% Enter this snr or psnr in the paper figure.

snr = @(x,y) 10 * log10( mean(y(:).^2) /  mean((x(:) - y(:)).^2));
for pattern_idx = 1:92
    meas = assort_meas(:, :, pattern_idx);
    target = assort_gt(:, :, pattern_idx);
    restored = assort_restored(:, :, pattern_idx);
    
    snr_all(1, pattern_idx) = snr(meas, target);
    snr_all(2, pattern_idx) = snr(restored, target);
end

psnr = @(x,y) 10 * log10(max(y(:))^2 /  mean((x(:) - y(:)).^2));
for pattern_idx = 1:92
    meas = assort_meas(:, :, pattern_idx);
    target = assort_gt(:, :, pattern_idx);
    restored = assort_restored(:, :, pattern_idx);
    
    psnr_all(1, pattern_idx) = psnr(meas, target);
    psnr_all(2, pattern_idx) = psnr(restored, target);
end
end