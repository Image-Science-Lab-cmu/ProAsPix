function [rsnr, psnr, ang_err] = get_performance_metric(x0, xhat)

x0 = max(0, x0);
x0 = x0/norm(x0(:));

xhat = max(0, xhat);
xhat = xhat/norm(xhat(:));



err = x0 - xhat;
rsnr = -20*log10(norm(err(:)));
psnr = 20*log10(sqrt(length(x0(:)))*max(x0(:))/norm(err(:)));

ang_err = sum(x0.*xhat, 3);
ang_err = ang_err./(1e-13+sqrt(sum(xhat.^2, 3)).*sqrt(sum(x0.^2, 3)));
ang_err = acos(ang_err)*180/pi;
ang_err = median(ang_err(:));

end