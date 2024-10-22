

srate = 10000;

t = 0:1/srate:2.7;
signal = zeros(size(t));


%% F

signal = signal + sin(2*pi*300*t)./(t-.7) + .8*sin(2*pi*200*t)./(t-.7);

tidx = dsearchn(t',[.67 .89]');
signal(tidx(1):tidx(2)) = signal(tidx(1):tidx(2)) + 200*sin(2*pi*324*t(tidx(1):tidx(2)));

tidx = dsearchn(t',[.65 .82]');
signal(tidx(1):tidx(2)) = signal(tidx(1):tidx(2)) + 200*sin(2*pi*200*t(tidx(1):tidx(2)));

%% u

tidx = dsearchn(t',[.9 1.05]');

freqTS = 50+170*(1-sqrt(abs(sin(2*pi*3*t(1:diff(tidx)+1)))));
centfreq = mean(freqTS);
k = (centfreq/srate)*2*pi/centfreq;
signal(tidx(1):tidx(2)) = signal(tidx(1):tidx(2)) + ...
    linspace(400,200,diff(tidx)+1).*sin(2*pi.*centfreq.*t(tidx(1):tidx(2)) + k*cumsum(freqTS-mean(freqTS)));

signal = signal + sin(2*pi*200*t)./(t-1.05) + .7*sin(2*pi*100*t)./(t-1.05);

%% n

tidx = dsearchn(t',[1.2 1.35]');

freqTS = 10+180*(sqrt(abs(sin(2*pi*2.25*t((1:diff(tidx)+1)+700)))));
centfreq = mean(freqTS);
k = (centfreq/srate)*2*pi/centfreq;
signal(tidx(1):tidx(2)) = signal(tidx(1):tidx(2)) + ...
    linspace(300,250,diff(tidx)+1).*sin(2*pi.*centfreq.*t(tidx(1):tidx(2)) + k*cumsum(freqTS-mean(freqTS)));

signal = signal + sin(2*pi*200*t)./(t-1.2) + .7*sin(2*pi*100*t)./(t-1.2) + 1.4*sin(2*pi*100*t)./(t-1.34);

%% T

signal = signal + sin(2*pi*300*t)./(t-1.6) + .8*sin(2*pi*200*t)./(t-1.6);

tidx = dsearchn(t',[1.45 1.75]');
signal(tidx(1):tidx(2)) = signal(tidx(1):tidx(2)) + 200*sin(2*pi*324*t(tidx(1):tidx(2)));

%% F

signal = signal + sin(2*pi*300*t)./(t-1.85) + .8*sin(2*pi*200*t)./(t-1.85);

tidx = dsearchn(t',[1.82 2.03]');
signal(tidx(1):tidx(2)) = signal(tidx(1):tidx(2)) + 200*sin(2*pi*324*t(tidx(1):tidx(2)));

tidx = dsearchn(t',[1.8 1.98]');
signal(tidx(1):tidx(2)) = signal(tidx(1):tidx(2)) + 200*sin(2*pi*200*t(tidx(1):tidx(2)));

%%

wherenan = find(~isfinite(signal));
for i=1:length(wherenan)
    signal(wherenan(i)) = mean(signal([wherenan(i)-1 wherenan(i)+1]));
end

clf
subplot(211)
plot(t,signal)
set(gca,'xlim',t([1 end]),'ylim',[-700 3100])

% setup wavelet convolution
wavetime = -2:1/srate:2;
Lconv = length(t)+length(wavetime)-1;
halfwavsize = floor(length(wavetime)/2);

nfrex = 50;
frex  = linspace(2,450,nfrex);
n     = logspace(log10(10),log10(20),nfrex);

% initialize
tf    = zeros(nfrex,length(t));
dataF = fft(signal,Lconv);

for fi=1:nfrex
    % compute and normalize wavelet
    w    = 2*( n(fi)/(2*pi*frex(fi)) )^2;
    cmwX = fft(exp(1i*2*pi*frex(fi).*wavetime) .* exp( (-wavetime.^2)/w ), Lconv);
    cmwX = cmwX./max(cmwX);
    
    % run convolution and extract power
    convres  = ifft( dataF .* cmwX );
    tf(fi,:) = abs(convres(halfwavsize:end-halfwavsize-1)).^2;
end

subplot(212)
contourf(t,frex,tf,50,'linecolor','none')
set(gca,'clim',[0 10000],'xlim',t([1 end]),'ylim',[frex(1) 450])

c = zeros(size(jet));
c(:,1) = linspace(0,.5,64);
c(:,2) = linspace(0,.5,64);
c(:,3) = linspace(0,1,64);
colormap(c)
axis off

%%

