

%% Chapter 11.1, Figure 11.1

% create signals A and B
srate = 1000;
t = 0:1/srate:9;
n = length(t);
f = [10 14 8];

k1=(mean(f)/srate)*2*pi/mean(f);

% sigA is a chirp plus noise
sigA = sin(2*pi.*f(1).*t + k1*cumsum(5*randn(size(t)))) + randn(size(t));

% sigB is sigA plus independent signal
sigB = sin(2*pi.*f(2).*t + k1*cumsum(5*randn(size(t)))) + sigA + randn(size(t));

% sigA gets its own independent signal added
sigA = sigA + sin(2*pi.*f(3).*t + k1*cumsum(5*randn(size(t))));

% Fourier transforms of sigA and sigB
hz    = linspace(0,srate/2,floor(n/2)+1);
sigAx = fft(sigA)/n;
sigBx = fft(sigB)/n;

% plot signals and their power spectra
clf
subplot(211)
plot(t,sigA), hold on
plot(t,sigB,'r')
xlabel('Time (s)'), ylabel('Amplitude')

subplot(223)
plot(hz,2*abs(sigAx(1:length(hz))))
hold on
plot(hz,2*abs(sigBx(1:length(hz))),'r')
set(gca,'xlim',[5 20],'ylim',[-.01 1.02])
xlabel('Frequency (Hz)'), ylabel('Amplitude')
legend({'sigA';'sigB'})

% compute spectral coherence
specX = abs(sigAx.*conj(sigBx)).^2;
spectcoher = specX./(sigAx.*sigBx);

subplot(224)
plot(hz,abs(spectcoher(1:length(hz))))
set(gca,'xlim',[5 20])
title('Spectral coherence')

%% Chapter 11.2, Figures 11.3/4/5

% covariance matrix
v = rand(10);
c = chol(v*v');

% n time points
n = 10000;
d = randn(n,size(v,1))*c;

% subtract mean
d = bsxfun(@minus,d,mean(d,1));
covar = (d'*d)./(n-1); % note: cov is a poor choice of variable name because it is a function

clf
subplot(311)
plot(d)
xlabel('Time (a.u.)'), ylabel('Amplitude')

subplot(323)
imagesc(covar), axis square
set(gca,'clim',[1 4])
colormap gray
title('Covariance')


% PCA via eigenvalue decomposition
[pc,eigvals] = eig(covar);

pc      = pc(:,end:-1:1); % resort PCs to go from largest to smallest
eigvals = diag(eigvals);
eigvals = 100*eigvals(end:-1:1)./sum(eigvals); % convert to percent change

subplot(324)
plot(eigvals,'o-','markerface','b')
title('Normalized eigenvalues')

% weight and plot time courses of first two PCs
subplot(313)
plot(pc(:,1)'*d')
hold on
plot(pc(:,2)'*d','r')
set(gca,'xlim',[0 100])
legend({'PC1';'PC2'})
xlabel('Time (a.u.)'), ylabel('PC amplitude')
title('Time course of first principal component')

%% end
