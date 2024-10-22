


%% Chapter 3, exercise 1

% 1) Generate a time series of square-wave-modulated sine waves,
%    such that the sine waves are present only when the square wave
%    is in the 'upper' state.

% There are several ways to produce such a wave.
% One is to create a stationary sine wave
%  and a square wave with values of 0 or 1.
% When the two signals are point-wise multiplied,
%  the result is the solution.

% first, the basics:
srate    = 1000;
time     = 0:1/srate:6;
sinefreq = 10; % in hz
boxfreq  =  1; % in hz

% second, create a sine wave
sinewave = sin(2*pi*sinefreq*time);

% third, create a box wave
boxwave  = sin(2*pi*boxfreq*time)>0;

% fourth, point-wise multiply the two signals
solution = boxwave .* sinewave;

% Although the above is technically a solution,
%  it would be nice to see the box better.
solution = solution + 3*boxwave;

% and plot the result
clf
plot(time,solution)
set(gca,'ylim',[-.5 4.5])

%% Chapter 3, exercise 2

% 2) Generate a time series by combining sine waves of several frequencies.
%    How many can you add together and still recognize the individual frequencies?


srate = 1000;
t = 0:1/srate:5;
a = [ 10   5   8   10 ];
f = [  3   6  12   .3 ];

sinewave = zeros(size(t));
for i=1:length(a)
    sinewave = sinewave + a(i)*sin(2*pi*f(i)*t);
end

plot(t,sinewave)
xlabel('Time (s)'), ylabel('amplitude')

%% Chapter 3, exercise 3

% 3) Generate a time series by combining three sine waves of different frequencies (all with constant amplitude of 1).
%    Make sure you can visually identify all three sine wave components.
%    Now add random noise. What is the (approximate) amplitude of noise such that the individual sine waves
%    are no longer visually recognizable?

srate = 1000;
t = 0:1/srate:5;
f = [  3   12   .3 ];

noisefactor = 1.5;

sinewave = zeros(size(t));
for i=1:length(f)
    sinewave = sinewave + sin(2*pi*f(i)*t);
end

sinewave = sinewave + randn(size(sinewave))*noisefactor;

plot(t,sinewave)
xlabel('Time (s)'), ylabel('amplitude')


%% Chapter 3, exercise 4

% 4) Generate two Gaussian time series, 
%    one with stationary noise added and 
%    one with non-stationary noise added.


clf

t = -1:.001:5;
w = [2 .5];
a = [4 5];


nonstatNoise = linspace(-5,10,length(t)) + randn(size(t)).*linspace(0,5,length(t));

g1 = a(1)*exp( (-t.^2) /(2*w(2)^2) );
g2 = a(2)*exp( (-(t-mean(t)).^2) /(2*w(2)^2) );

g1 = g1+randn(size(g1));
g2 = g2+nonstatNoise;

plot(t,g2,'r'), hold on
plot(t,g1)
hold off

%% Chapter 3, exercise 5

% 5) Create two overlapping sinc functions of different frequencies and different peak latencies. 
% A sinc function generally takes the form of sin(t)/t, and can be expanded to frequency and peak-timing 
% specificity using, in Matlab code, sin(2*pi*f*(t-m))./(t-m), where f is the peak frequency, t is a vector 
% of time indices in milliseconds, and m is the peak time of the sinc function. Note that when t-m is zero, 
% the function will have an NaN value (not-a-number). This can cause difficulties in frequency and time-frequency 
% analyses, and can be replaced with the average of surrounding time points.

srate=1000;
t=-3:1/srate:3;
m=[-1 .5];
f=[5 12];

signal = sin(2*pi*f(1)*(t-m(1)))./(t-m(1)) + sin(2*pi*f(2)*(t-m(2)))./(t-m(2));

wherenan = find(~isfinite(signal));
for i=1:length(wherenan)
    signal(wherenan(i)) = mean(signal([wherenan(i)-1 wherenan(i)+1]));
end

clf
plot(t,signal)
xlabel('Time (s)'), ylabel('Amplitude')

%%



%% Chapter 4, exercise 1

% first, the basics:
srate = 1000;
time  = 0:1/srate:6;
N     = length(time);
f     = [1 5]; % in hz
ff    = linspace(f(1),f(2)*mean(f)/f(2),length(time));
data  = sin(2*pi.*ff.*time);

% second, compute Fourier transform
dataX = fft(data);

% third, shuffle phases
phases = angle([dataX(10:end) dataX(1:9)]);
shuffdata = abs(dataX).*exp(1i*phases);

% fourth, reconstruct the signal
newdata  = ifft(shuffdata);
newdataX = fft(newdata);

% fifth, plot the results
clf
subplot(211)
plot(time,data), hold on
plot(time,real(newdata),'r')

subplot(212)
hz = linspace(0,srate/2,floor(N/2)+1);
plot(hz,2*abs(dataX(1:length(hz))/N),'o'), hold on
plot(hz,2*abs(newdataX(1:length(hz))/N),'r')
set(gca,'xlim',[0 20])

%% Chapter 4, exercise 2

% 2) Reproduce Figure 4.15 (section 4.10) four additional times, setting the amplitude of the random noise to be 10 or 100, and the number of trials to be 100 or 1000. 
% Are the individual peak frequencies still clearly visible in the power plot? What does your answer indicate about collecting more data with a lot of 
% noise, vs. less data with less noise?

figure(1), clf

srate   = 1000;
t       = 0:1/srate:5; 
npnts   = length(t);

a = [2 3 4 2];
f = [1 3 6 12];

data = zeros(size(t));

for i=1:length(a)
    data = data + a(i)*sin(2*pi*f(i)*t);
end

hz = linspace(0,srate/2,floor(npnts/2)+1);
hanwin = .5*(1-cos(2*pi*linspace(0,1,npnts)));


for i=1:4
    
    nTrials   = 10 + mod(i+1,2)*990;
    noiseFact = 10 + (i>2)*90;
    
    x = 2*abs(fft(bsxfun(@times,bsxfun(@plus,data,noiseFact*randn(nTrials,npnts)),hanwin),[],2)/npnts);
    subplot(2,2,i), plot(hz,mean(x(:,1:length(hz))),'k')
    set(gca,'xlim',[0 20],'ylim',[0 2.5])
    title([ num2str(nTrials) ' trials, ' num2str(noiseFact) 'X noise' ])
end

%% Chapter 4, exercise 3

% 3) Generate a 2-second sine wave of 10 Hz. Compute its Fourier transform and plot the 
% resulting power spectrum (do not apply a taper). Next, re-compute the Fourier transform 
% with 2 additional seconds of zero padded data. Instead of scaling the Fourier coefficients 
% by the number of time points in the original sine wave, however, scale the coefficients by 
% the number of time points including the zero padding (thus, the number of time points in 4 
% seconds). What do you notice about the amplitude between the two analyses? What happens to 
% the amplitude if there are 4 seconds of zero padding, or 1 second of zero padding? Explain 
% why these differences occur.

clf

srate = 1000;
t = 0:1/srate:2;
signal = sin(2*pi*10*t);


for i=1:4
    subplot(2,2,i)
    
    if     i==1, N=length(t);
    elseif i==2, N=length(t)+2*srate;
    elseif i==3, N=length(t)+4*srate;
    elseif i==4, N=length(t)+1*srate;
    end
    
    x  = fft(signal,N)/N;
    hz = linspace(0,srate/2,floor(N/2)+1);
    bar(hz,2*abs(x(1:length(hz))))
    set(gca,'xlim',[5 15],'ylim',[0 1.02])
    
    title([ 'Nfft = ' num2str(N) ])
    legend([ 'Peak power = ' num2str(round(100*max(2*abs(x)))/100) ])
end

%%



%% Chapter 5, exercise 1

clf, colormap gray

% first, create chirp
srate = 100;
t     = 0:1/srate:5;
f     = [1 40];
ff    = linspace(f(1),f(2)*mean(f)/f(2),length(t));
signal= sin(2*pi.*ff.*t);

subplot(311)
plot(t,signal)
set(gca,'ylim',[-1.1 1.1])

% list of window widths (in ms)
FFTwidths  = [50 100 200 500];
Ntimesteps = 20; % number of time widths

for widi=1:length(FFTwidths)
    
    % define window width and convert to time steps
    fftWidth_ms = FFTwidths(widi);
    fftWidth    = round(fftWidth_ms/(1000/srate)/2);
    centertimes = round(linspace(fftWidth+1,length(t)-fftWidth,Ntimesteps));
    
    % initialize parameters
    hz = linspace(0,srate/2,fftWidth-1);
    tf = zeros(length(hz),length(centertimes));
    hanwin = .5*(1-cos(2*pi*(1:fftWidth*2)/(fftWidth*2-1)));
    
    % loop through center time points and compute FFT
    for ti=1:length(centertimes)
        temp = signal(centertimes(ti)-fftWidth:centertimes(ti)+fftWidth-1);
        x = fft(hanwin.*temp)/fftWidth*2;
        tf(:,ti) = 2*abs(x(1:length(hz)));
    end
    
    % and plot
    subplot(3,2,widi+2)
    contourf(t(centertimes),hz,tf,1)
    set(gca,'ylim',[0 40],'clim',[0 1],'xlim',t([1 end]))
    xlabel('Time (s)'), ylabel('Frequency (Hz)')
    title([ 'FFT width: ' num2str(FFTwidths(widi)) ' ms' ])
end

%% Chapter 5, exercise 2

% 2) Recompute figure 5.1, but adapt the time-frequency analysis so that it captures 
% the dynamics in the first and last 500 ms.

srate = 1000; 
t = 0:1/srate:5;

f = [30 3 6 12];
timechunks = round(linspace(1,length(t),length(f)+1));


data = 0;
for i=1:length(f)
    data = cat(2,data,sin(2*pi*f(i)*t(timechunks(i):timechunks(i+1)-1) ));
end

figure(1),clf

% FFT parameters
fftWidth_ms = 1000;
fftWidth=round(fftWidth_ms/(1000/srate)/2);
Ntimesteps = 10; % number of time widths
centertimes = round(linspace(1,length(t),Ntimesteps));

hz=linspace(0,srate/2,fftWidth-1);
tf=zeros(length(hz),length(centertimes));
hanwin = .5*(1-cos(2*pi*(1:fftWidth*2)/(fftWidth*2-1)));

for ti=1:length(centertimes)
    
    if ti==1
        temp = [ data(centertimes(ti)+fftWidth-1:-1:centertimes(ti)) data(centertimes(ti):centertimes(ti)+fftWidth-1) ];
    elseif ti==length(centertimes)
        temp = [ data(centertimes(ti)-fftWidth:centertimes(ti)) data(centertimes(ti)-fftWidth+2:centertimes(ti)) ];
    else
        temp = data(centertimes(ti)-fftWidth:centertimes(ti)+fftWidth-1);
    end
    
    x = fft(hanwin.*temp)/fftWidth*2;
    tf(:,ti) = 2*abs(x(1:length(hz)));
end

subplot(212)
contourf(t(centertimes),hz,tf,1)
set(gca,'ylim',[0 40],'clim',[0 1],'xlim',[0 5])
xlabel('Time (s)'), ylabel('Frequency (Hz)')

colormap gray

%% Chapter 5, exercise 3

% 3) Generate a 10-second signal with 5 seconds at one frequency and 5 seconds at another frequency 
% (similar to figure 5.1 but for two frequencies). Compute the time-frequency representation of the 
% signal, and also the 'static' Fourier transform of the entire signal as shown in chapter 4. From 
% the time-frequency representation, sum the frequency values over all time points to obtain a power 
% spectrum plot like that of the 'static' Fourier transform. How do the amplitudes compare between 
% the two analyses?

srate = 100;
t = 0:1/srate:9;
f = [ 10 20 ];

signal = [ sin(2*pi*f(1)*t(1:floor(length(t)/2))) sin(2*pi*f(2)*t(ceil(length(t)/2):end)) ];

% static FFT
hanwin    = .5*(1-cos(2*pi*(1:length(t))/(length(t)-1)));
staticFFT = 2*abs(fft(hanwin.*signal)/length(t));
staticHz  = linspace(0,srate/2,floor(length(t)/2)+1);

clf
subplot(211)
plot(t,hanwin.*signal)

% short-time FFT
fftWidth_ms = 500;
% note! Try changing the width above and re-compare to the static FFT


fftWidth = round(fftWidth_ms/(1000/srate)/2);
Ntimesteps = 30; % number of time widths
centertimes = round(linspace(fftWidth+1,length(t)-fftWidth,Ntimesteps));

tfhz   = linspace(0,srate/2,fftWidth+1);
tf     = zeros(length(tfhz),length(centertimes));
hanwin = .5*(1-cos(2*pi*(1:fftWidth*2)/(fftWidth*2-1)));

for ti=1:length(centertimes)
    temp = signal(centertimes(ti)-fftWidth:centertimes(ti)+fftWidth-1);
    x = fft(hanwin.*temp)/(fftWidth*2);
    tf(:,ti) = 2*abs(x(1:length(tfhz)));
end

subplot(223)
contourf(t(centertimes),tfhz,tf,40,'linecolor','none')

subplot(224)
plot(staticHz,staticFFT(1:length(staticHz)),'.-')
hold on
plot(tfhz,mean(tf,2),'r.-')

%%



%% Chapter 6, exercise 1

clf

% The signal will have two overlapping sine waves,
% one with increasing power and one with decreasing power.

% zeroth, the basics:
srate     = 1000;
time      = 0:1/srate:6;
sinefreqs = [5 11]; % in hz
numTrials = 20;
wavetime  = -2:1/srate:2;
nfrex     = 9; 
frex      = logspace(log10(2),log10(30),nfrex);
n         = logspace(log10(3),log10(12),nfrex);
Lconv     = length(time)+length(wavetime)-1;
halfwavsize= floor(length(wavetime)/2);

% initialize matrices
signal = zeros(numTrials,length(time));
tf     = zeros(2,nfrex,length(time));

% first, create the signal
ampInc = linspace(0,1,length(time));
ampDec = linspace(1,0,length(time));

for ti=1:numTrials
    sine1 = ampInc.*sin(2*pi*sinefreqs(1)*time + rand*2*pi);
    sine2 = ampDec.*sin(2*pi*sinefreqs(2)*time + rand*2*pi);
    signal(ti,:) =  sine1+sine2;
end

% second, perform time-frequency decomposition
signalX1 = fft(signal,Lconv,2);     % FFT along 2nd dimension
signalX2 = fft(mean(signal),Lconv); % note that trials are first averaged

for fi=1:nfrex
    % create wavelet
    w     = 2*( n(fi)/(2*pi*frex(fi)) )^2; % w is width of Gaussian
    cmwX  = fft(exp(1i*2*pi*frex(fi).*wavetime) .* exp( (-wavetime.^2)/w ), Lconv);
    cmwX  = cmwX./max(cmwX);
    
    % convolution of all trials simultaneously using bxsfun
    convres    = ifft( bsxfun(@times,signalX1,cmwX) ,[],2);
    temppower  = 2*abs(convres(:,halfwavsize:end-halfwavsize-1));
    tf(1,fi,:) = mean(temppower,1);
    
    % The 2 lines above are equivalent to the 5 lines below, which uses
    % loops and thus should be avoided when possible.
%     for ti=1:numTrials
%         dataX = fft(signal(ti,:),Lconv);
%         convres = ifft( dataX .* mwave_fft );
%         tf(ti,fi,:) = 2*abs(convres(halfwavsize:end-halfwavsize-1));
%     end

    
    % convolution of trial average (third)
    convres    = ifft( signalX2.*cmwX );
    tf(2,fi,:) = 2*abs(convres(:,halfwavsize:end-halfwavsize-1));
end

% finally (fourth), plot the average and TF power
subplot(221)
plot(time,signal)
set(gca,'ylim',[-1.5 1.5])

subplot(223)
contourf(time,frex,squeeze(tf(1,:,:)),40,'linecolor','none')
set(gca,'clim',[0 .6])
colormap gray


% 
subplot(222)
plot(time,mean(signal))
set(gca,'ylim',[-1.5 1.5])

subplot(224)
contourf(time,frex,squeeze(tf(2,:,:)),40,'linecolor','none')
set(gca,'clim',[0 .6])
colormap gray

%% Chapter 6, exercise 2

% 2) Generate a 10-second time series that comprises the following two sine chirps: 
% One lasting from 2 to 7 seconds and increasing in frequency from 10 to 25 Hz while 
% also increasing in power, and one lasting from 5 to 10 seconds and decreasing in 
% frequency from 45 to 30 Hz while also decreasing in power. Perform a time-frequency 
% analysis of this signal using complex Morlet wavelet convolution. Justify your choice 
% of parameter selection (number and range of wavelet frequencies, and their Gaussian widths).


doexercise3 = 1;

srate = 1000;
t = 0:1/srate:10;
signal = zeros(size(t));

% create first part of signal
tidx     = dsearchn(t',[2 7]');
freqTS   = linspace(10,25,length(t(tidx(1):tidx(2)))); 
centfreq = mean(freqTS);
k        = (centfreq/srate)*2*pi/centfreq;
pow      = linspace(1,3,length(t(tidx(1):tidx(2))));

signal(tidx(1):tidx(2)) = pow.*sin(2*pi.*centfreq.*t(tidx(1):tidx(2)) + k*cumsum(freqTS-centfreq));


% create second part of signal
tidx     = dsearchn(t',[5 10]');
freqTS   = linspace(45,30,length(t(tidx(1):tidx(2)))); 
centfreq = mean(freqTS);
k        = (centfreq/srate)*2*pi/centfreq;
pow      = linspace(4,2,length(t(tidx(1):tidx(2))));

signal(tidx(1):tidx(2)) = signal(tidx(1):tidx(2)) + pow.*sin(2*pi.*centfreq.*t(tidx(1):tidx(2)) + k*cumsum(freqTS-centfreq));


% for exercise 3
if doexercise3
    signal = signal + 5*real(ifft( fft(randn(size(t))) .* linspace(-1,1,length(t)).^2 ));
end



clf, colormap hot
subplot(211)
plot(t,signal)


% setup wavelet convolution
wavetime = -2:1/srate:2;
Lconv = length(t)+length(wavetime)-1;
halfwavsize = floor(length(wavetime)/2);

nfrex = 30;
frex  = logspace(log10(2),log10(60),nfrex);
n     = logspace(log10(3),log10(15),nfrex);
% note: try changing the parameters of 'n' to, e.g., 3 to 5


% initialize
tf    = zeros(nfrex,length(t));
dataF = fft(signal,Lconv);

for fi=1:nfrex
    % compute and normalize wavelet
    w         = 2*( n(fi)/(2*pi*frex(fi)) )^2;
    cmwX = fft(exp(1i*2*pi*frex(fi).*wavetime) .* exp( (-wavetime.^2)/w ), Lconv);
    cmwX = cmwX./max(cmwX);
    
    % run convolution and extract power
    convres  = ifft( dataF .* cmwX );
    tf(fi,:) = 2*abs(convres(halfwavsize:end-halfwavsize-1));
end

subplot(212)
contourf(t,frex,tf,40,'linecolor','none')
set(gca,'clim',[0 2]), colorbar

%% Chapter 6, exercise 3

% 3) Take the signal used in exercise 2, and add pink noise with 1/f characteristics. 
% The maximum amplitude of the pink noise should be around twice the average amplitude 
% of the noise-less signal. Re-do the time-frequency analysis. Does the noise have a stronger 
% effect on the lower frequency or the higher frequency chirp, or roughly the same effect?

% see above...


%% Chapter 6, exercise 4

% 4) Re-do exercises 1 (analysis of single trials and then average, not the analysis of the 
% trial-average) and 2 above, except using the short-time Fourier transform instead of Morlet 
% wavelets. Plot the results of the two analyses (wavelet, short-time Fourier) next to each other. 
% Based on visual inspection, are there any noticeable qualitative differences between the two 
% approaches? Does one seem better than the other, and does this depend on the signal characteristics?

redo_exercise1 = 0; % true for 1, false for 2


if redo_exercise1
    
    % The signal will have two overlapping sine waves,
    % one with increasing power and one with decreasing power.
    
    % first, the basics:
    srate     = 1000;
    time      = 0:1/srate:6;
    sinefreqs = [5 11]; % in hz
    numTrials = 20;
    wavetime  = -2:1/srate:2;
    nfrex     = 50;
    frex      = logspace(log10(2),log10(30),nfrex);
    n         = logspace(log10(10),log10(25),nfrex);
    Lconv     = length(time)+length(wavetime)-1;
    halfwavsize= floor(length(wavetime)/2);
    
    % second, initialize matrices
    signal = zeros(numTrials,length(time));
    tf     = zeros(nfrex,length(time));
    
    % third, create the signal
    ampInc = linspace(0,1,length(time));
    ampDec = linspace(1,0,length(time));
    
    for ti=1:numTrials
        sine1 = ampInc.*sin(2*pi*sinefreqs(1)*time + rand*2*pi);
        sine2 = ampDec.*sin(2*pi*sinefreqs(2)*time + rand*2*pi);
        signal(ti,:) =  sine1+sine2;
    end
    
    % fourth, perform time-frequency decomposition
    dataX = fft(signal,Lconv,2);     % FFT along 2nd dimension
    
    for fi=1:nfrex
        % create wavelet
        w          = 2*( n(fi)/(2*pi*frex(fi)) )^2; % w is width of Gaussian
        cmwX  = fft(exp(1i*2*pi*frex(fi).*wavetime) .* exp( (-wavetime.^2)/w ), Lconv);
        cmwX  = cmwX./max(cmwX);
        
        % convolution of all trials simultaneously using bxsfun
        convres    = ifft( bsxfun(@times,dataX,cmwX) ,[],2);
        temppower  = 2*abs(convres(:,halfwavsize:end-halfwavsize-1));
        tf(fi,:) = mean(temppower,1);
    end

    clf
    subplot(221)
    contourf(time,frex,tf,40,'linecolor','none')
    set(gca,'clim',[0 .6],'xlim',time([1 end]))
    colormap gray
    
    
    
    
    
    % short-time FFT
    fftWidth_ms = 1000;
    % note! Try changing the width above and re-compare to the static FFT
    
    fftWidth = round(fftWidth_ms/(1000/srate)/2);
    Ntimesteps = 30; % number of time widths
    centertimes = round(linspace(fftWidth+1,length(time)-fftWidth,Ntimesteps));
    
    tfhz   = linspace(0,srate/2,fftWidth+1);
    tf     = zeros(length(tfhz),length(centertimes));
    hanwin = .5*(1-cos(2*pi*(1:fftWidth*2)/(fftWidth*2-1)));
    
    for ti=1:length(centertimes)
        temp = signal(:,centertimes(ti)-fftWidth:centertimes(ti)+fftWidth-1);
        x = fft(bsxfun(@times,hanwin,temp),[],2)/(fftWidth*2);
        tf(:,ti) = 2*mean(abs(x(:,1:length(tfhz))));
    end
    
    subplot(223)
    contourf(time(centertimes),tfhz,tf,40,'linecolor','none')
    set(gca,'ylim',[0 30],'clim',[0 .6],'xlim',time([1 end]))
else
    
    
    srate = 1000;
    t = 1:1/srate:10;
    signal = zeros(size(t));
    
    % create first part of signal
    tidx     = dsearchn(t',[2 7]');
    freqTS   = linspace(10,25,length(t(tidx(1):tidx(2))));
    centfreq = mean(freqTS);
    k        = (centfreq/srate)*2*pi/centfreq;
    pow      = linspace(1,3,length(t(tidx(1):tidx(2))));
    
    signal(tidx(1):tidx(2)) = pow.*sin(2*pi.*centfreq.*t(tidx(1):tidx(2)) + k*cumsum(freqTS-centfreq));
    
    
    % create second part of signal
    tidx     = dsearchn(t',[5 10]');
    freqTS   = linspace(45,30,length(t(tidx(1):tidx(2))));
    centfreq = mean(freqTS);
    k        = (centfreq/srate)*2*pi/centfreq;
    pow      = linspace(4,2,length(t(tidx(1):tidx(2))));
    
    signal(tidx(1):tidx(2)) = signal(tidx(1):tidx(2)) + pow.*sin(2*pi.*centfreq.*t(tidx(1):tidx(2)) + k*cumsum(freqTS-centfreq));
    
    % setup wavelet convolution
    wavetime = -2:1/srate:2;
    Lconv = length(t)+length(wavetime)-1;
    halfwavsize = floor(length(wavetime)/2);
    
    nfrex = 60;
    frex  = logspace(log10(2),log10(60),nfrex);
    n     = logspace(log10(5),log10(25),nfrex);
    % note: try changing the parameters of 'n' to, e.g., 3 to 5
    
    
    % initialize
    tf    = zeros(nfrex,length(t));
    dataF = fft(signal,Lconv);
    
    for fi=1:nfrex
        % compute and normalize wavelet
        w         = 2*( n(fi)/(2*pi*frex(fi)) )^2;
        cmwX = fft(exp(1i*2*pi*frex(fi).*wavetime) .* exp( (-wavetime.^2)/w ), Lconv);
        cmwX = cmwX./max(cmwX);
        
        % run convolution and extract power
        convres  = ifft( dataF .* cmwX );
        tf(fi,:) = 2*abs(convres(halfwavsize:end-halfwavsize-1));
    end
    
    subplot(222)
    contourf(t,frex,tf,40,'linecolor','none')
    set(gca,'clim',[0 2],'xlim',t([1 end]))
    
    
    
    
    
    
    % short-time FFT
    fftWidth_ms = 500;
    % note! Try changing the width above and re-compare to the static FFT
    
    fftWidth = round(fftWidth_ms/(1000/srate)/2);
    Ntimesteps = 30; % number of time widths
    centertimes = round(linspace(fftWidth+1,length(t)-fftWidth,Ntimesteps));
    
    tfhz   = linspace(0,srate/2,fftWidth+1);
    tf     = zeros(length(tfhz),length(centertimes));
    hanwin = .5*(1-cos(2*pi*(1:fftWidth*2)/(fftWidth*2-1)));
    
    for ti=1:length(centertimes)
        temp = signal(centertimes(ti)-fftWidth:centertimes(ti)+fftWidth-1);
        x = fft(hanwin.*temp)/(fftWidth*2);
        tf(:,ti) = 2*abs(x(1:length(tfhz)));
    end
    
    subplot(224)
    contourf(t(centertimes),tfhz,tf,40,'linecolor','none')
    set(gca,'ylim',[0 60],'clim',[0 .6],'xlim',t([1 end]))
end

%%





%% Chapter 7, exercise 1

% 1) Generate a signal that increases from 5 Hz to 15 Hz over 5 seconds, with randomly varying amplitude. 
% Convolve this signal with a 10-Hz complex Morlet wavelet, and, separately apply the filter-Hilbert method 
% using a plateau-shaped band-pass filter centered at 10 Hz. From both results, extract the filtered signal 
% and the power, and plot them simultaneously. Do the results from these two methods look similar or different, 
% and how could they be made to look more similar or more different?

srate   = 1000;
nyquist = srate/2;
t       = 0:1/srate:4;

% create first part of signal
freqTS   = linspace(5,15,length(t));
centfreq = mean(freqTS);
k        = (centfreq/srate)*2*pi/centfreq;
pow      = abs(interp1(linspace(t(1),t(end),10),10*rand(1,10),t,'spline'));
signal   = pow.*sin(2*pi.*centfreq.*t + k*cumsum(freqTS-centfreq));



% wavelet convolution
wavetime = -2:1/srate:2;
Lconv = length(t)+length(wavetime)-1;
halfwavsize = floor(length(wavetime)/2);
% compute and normalize wavelet
w         = 2*( 6/(2*pi*10) )^2;
cmwX = fft(exp(1i*2*pi*10.*wavetime) .* exp( (-wavetime.^2)/w ), Lconv);
cmwX = cmwX./max(cmwX);
% run convolution
convres = ifft( fft(signal,Lconv) .* cmwX );
convres = convres(halfwavsize:end-halfwavsize-1);



% band-pass filter
band   = [9.5 10.5]; % Hz 
twidth = 0.15;
filtO  = round(3*(srate/band(1))); 
freqs  = [ 0 (1-twidth)*band(1) band(1) band(2) (1+twidth)*band(2) nyquist ]/nyquist;
idealresponse = [ 0 0 1 1 0 0 ]; 
filterweights = firls(filtO,freqs,idealresponse); 
filtered_data = filtfilt(filterweights,1,signal);

clf
subplot(311), plot(t,signal)
subplot(312), plot(t,real(convres)),  hold on, plot(t,filtered_data,'r')
subplot(313), plot(t,2*abs(convres)), hold on, plot(t,2*abs(hilbert(filtered_data)),'r')

%% Chapter 7, exercise 2

% 2) Reproduce the time-frequency power plot in figure 5.1 (the signal comprises concatenated sine waves at 
% 30, 3, 6, and 12 Hz over 6 seconds with a sampling rate of 1000 Hz), but using the filter-Hilbert method. 
% To do this exercise, it is necessary to filter the signal in multiple overlapping frequency bands. How do the 
% results compare to that shown in Figure 5.1? Is it now possible to estimate power before .5 seconds and after 
% 4.5 seconds with any zero-padding or reflection? Why or why not?

srate = 1000; 
nyquist = srate/2;
t     = 0:1/srate:5;
f     = [30 3 6 12];

timechunks = round(linspace(1,length(t),length(f)+1));

data = 0;
for i=1:length(f)
    data = cat(2,data,sin(2*pi*f(i)*t(timechunks(i):timechunks(i+1)-1) ));
end


clf
subplot(211)
plot(t,data)


nfrex = 30;
centerfreqs = linspace(2,40,nfrex);
tf = zeros(nfrex,length(t));
n = linspace(3,8,nfrex);

for fi=1:nfrex
    
    % band-pass filter
    band   = [centerfreqs(fi)-1 centerfreqs(fi)+1];
    twidth = 0.1;
    filtO  = round(n(fi)*(srate/band(1)));
    freqs  = [ 0 (1-twidth)*band(1) band(1) band(2) (1+twidth)*band(2) nyquist ]/nyquist;
    iresp  = [ 0 0 1 1 0 0 ];
    fweights = firls(filtO,freqs,iresp);
    
    filtdat = filtfilt(fweights,1,[data(end:-1:1) data data(end:-1:1)]);
    filtdat = filtdat(length(t)+1:length(t)*2);
    
    tf(fi,:) = 2*abs(hilbert(filtdat));
end

% plot TF
subplot(212)
contourf(t,centerfreqs,tf,1)
set(gca,'clim',[0 1],'xlim',[0 5])
xlabel('Time (s)'), ylabel('Frequency (Hz)')

colormap gray

%% Chapter 7, exercise 3

% 3) With Morlet wavelet convolution, the trade-off between temporal precision and frequency resolution 
% is defined by the width of the Gaussian that tapers the sine wave; in band-pass filtering, this trade-off 
% is defined by the width of the filter in the frequency domain. Explore this parameter by performing 
% time-frequency decomposition on a chirp (ranging from 20-80 Hz over 7 seconds with a sampling rate of 1000 Hz) 
% using four sets of parameters: Constant narrow-band filters, constant wide-band filters, filters with widths 
% that increase as a function of frequency, and filters with widths that decrease as a function of frequency. 
% Note that there is a limit as to how wide the filters can be at lower frequencies (it is not possible to have 
% a filter with 10-Hz width and a 3 Hz center). Are there noticeable differences amongst these settings, and 
% in which circumstances would it be best to use each parameter setting?

srate   = 1000; 
nyquist = srate/2;
t       = 0:1/srate:6; 
f       = [20 80];
chirpTS = sin(2*pi.*linspace(f(1),f(2)*mean(f)/f(2),length(t)).*t);


nfrex = 30;
centerfreqs = linspace(10,100,nfrex);
tf = zeros(nfrex,length(t));
n = linspace(5,30,nfrex);

bandwidth{1} = 1*ones(nfrex,1);
bandwidth{2} = 5*ones(nfrex,1);
bandwidth{3} = linspace(.5,20,nfrex);
bandwidth{4} = linspace(7,.01,nfrex);


for parami=1:4
    for fi=1:nfrex
        
        % band-pass filter
        band   = [centerfreqs(fi)-bandwidth{parami}(fi) centerfreqs(fi)+bandwidth{parami}(fi)];
        twidth = 0.1;
        filtO  = round(n(fi)*(srate/band(1)));
        freqs  = [ 0 (1-twidth)*band(1) band(1) band(2) (1+twidth)*band(2) nyquist ]/nyquist;
        iresp  = [ 0 0 1 1 0 0 ];
        fweights = firls(filtO,freqs,iresp);
        
        filtdat = filtfilt(fweights,1,[chirpTS(end:-1:1) chirpTS chirpTS(end:-1:1)]);
        filtdat = filtdat(length(t)+1:length(t)*2);
        
        tf(fi,:) = 2*abs(hilbert(filtdat));
    end
    
    % plot TF
    subplot(2,2,parami)
    contourf(t,centerfreqs,tf,1)
    set(gca,'clim',[0 1],'xlim',[0 6])
    xlabel('Time (s)'), ylabel('Frequency (Hz)')
end

colormap gray

%%






%% Chapter 8, exercise 1

% 1) Create a signal with multiple frequency non-stationarities that are contained within narrow ranges, 
% that is, non-stationarities between 10 and 13 Hz, and between 30 and 35 Hz). Compute instantaneous 
% frequency on the broadband signal. Next, apply appropriate band-pass filtering to isolate the two 
% frequency ranges, and compute instantaneous frequency again in these two frequency windows. Are the 
% results from the broadband analysis interpretable? Finally, perform a time-frequency analysis on these 
% data via Morlet wavelet convolution and plot the time-frequency power map. To what extent (if any) are 
% the frequency non-stationarities visible in the plot? What does this indicate about detecting relatively 
% subtle changes in time-varying frequencies using when assuming local frequency stationarity (Morlet 
% wavelet convolution) vs. not assuming stationarity.

% create signal
srate = 1000;
t = 0:1/srate:6;

freqTS = linspace(10,13,length(t));
centfreq = mean(freqTS);
k = (centfreq/srate)*2*pi/centfreq;
signal = sin(2*pi.*centfreq.*t + k*cumsum(freqTS-centfreq));

freqTS = linspace(35,30,length(t));
centfreq = mean(freqTS);
k = (centfreq/srate)*2*pi/centfreq;
signal = signal + sin(2*pi.*centfreq.*t + k*cumsum(freqTS-centfreq));


% instantaneous frequency of broadband signal
is_broadband = srate*diff(unwrap(angle(hilbert(signal))))/(2*pi);
is_broadband(length(t)) = is_broadband(end);

clf
subplot(221), plot(t,is_broadband)
set(gca,'ylim',[5 40])

% filter at 11.5 and 32.5 Hz and compute instantaneous frequency again
wavetime  = -2:1/srate:2;
Lconv     = length(t)+length(wavetime)-1;
halfwavsize= floor(length(wavetime)/2);
frex = [11.5 32.5];

sigX = fft(signal,Lconv);
is   = zeros(length(frex),length(t)-1);

for fi=1:2
    w          = 2*( 10/(2*pi*frex(fi)) )^2; % w is width of Gaussian
    cmwX  = fft(exp(1i*2*pi*frex(fi).*wavetime) .* exp( (-wavetime.^2)/w ), Lconv);
    cmwX  = cmwX./max(cmwX);
    convres  = ifft( sigX.*cmwX );
    convres = convres(halfwavsize:end-halfwavsize-1);
    
    is(fi,:) = srate*diff(unwrap(angle(convres)))/(2*pi);
end
is(:,length(t)) = is(:,end);

subplot(222)
plot(t,is)
set(gca,'ylim',[5 40])



% TF analysis with Morlet wavelet convolution
nfrex = 50;
frex  = linspace(5,40,nfrex);
tf    = zeros(length(frex),length(t));
n = linspace(8,30,nfrex);

for fi=1:nfrex
    w          = 2*( n(fi)/(2*pi*frex(fi)) )^2; % w is width of Gaussian
    cmwX  = fft(exp(1i*2*pi*frex(fi).*wavetime) .* exp( (-wavetime.^2)/w ), Lconv);
    cmwX  = cmwX./max(cmwX);
    convres  = ifft( sigX.*cmwX );
    convres = convres(halfwavsize:end-halfwavsize-1);
    
    tf(fi,:) = abs(convres).^2;
end

subplot(223)
contourf(t,frex,tf,40,'linecolor','none')


subplot(224)
freqrange = dsearchn(frex',[25 50]');
[vals,idx] = max(tf(freqrange(1):freqrange(2),:));
plot(t,frex(idx+freqrange(1)+1))
hold on

freqrange = dsearchn(frex',[4 25]');
[vals,idx] = max(tf(freqrange(1):freqrange(2),:));
plot(t,frex(idx+freqrange(1)+1))
set(gca,'ylim',[5 40])

%% chapter 8, exercise 2

srate=1000; t=0:1/srate:5;
sinewave=zeros(size(t));

% pairs of frequencies to use
fr = [13 16; 24.5 24; 32 39; 50 48];

for i=1:4
    ff = linspace(fr(i,1),fr(i,2)*mean(fr(i,:))/fr(i,2),length(t));
    sinewave = sinewave + sin(2*pi.*ff.*t);
end

hz   = linspace(0,srate/2,floor(length(t)/2)+1);
x    = fft(sinewave)/length(t);
imfs = emdx(sinewave,4);
f    = fft(imfs,[],2)/length(t);

figure(1),clf
subplot(311), plot(t,sinewave)
subplot(312), plot(hz,2*abs(x(1:length(hz))))
 set(gca,'xlim',[0 60])
subplot(313)
plot(hz,abs(f(:,1:length(hz))).^2,'linew',2)
 set(gca,'xlim',[0 60])

%
figure(4),clf

instfreqs = zeros(4,length(t)-1);
for i=1:4%size(imfsx,1)
    instfreqs(i,:) = srate*diff(unwrap(angle(hilbert(imfs(i,:)))))/(2*pi);
end
instfreqs(:,length(t)) = instfreqs(:,end);

clf
subplot(311)
plot(t,instfreqs,'.','markersize',1)
set(gca,'ylim',[0 60])

%

subplot(312)

% first, the basics:
wavetime   = -2:1/srate:2;
nfrex      = 50;
frex       = logspace(log10(2),log10(60),nfrex);
n          = logspace(log10(10),log10(20),nfrex);
Lconv      = length(t)+length(wavetime)-1;
halfwavsize= floor(length(wavetime)/2);

% second, initialize matrices
tf = zeros(nfrex,length(t));

% fourth, perform time-frequency decomposition
dataX = fft(sinewave,Lconv);     % FFT along 2nd dimension

for fi=1:nfrex
    % create wavelet
    w          = 2*( n(fi)/(2*pi*frex(fi)) )^2; % w is width of Gaussian
    cmwX  = fft(exp(1i*2*pi*frex(fi).*wavetime) .* exp( (-wavetime.^2)/w ), Lconv);
    cmwX  = cmwX./max(cmwX);
    
    % convolution of all trials simultaneously using bxsfun
    convres    = ifft( bsxfun(@times,dataX,cmwX) ,[],2);
    temppower  = 2*abs(convres(:,halfwavsize:end-halfwavsize-1));
    tf(fi,:) = mean(temppower,1);
end

contourf(t,frex,tf,40,'linecolor','none')
% colormap gray

%


subplot(313)
hold all

% define frequencies
nfrex = size(fr,1);
frex  = mean(fr,2);

for fi=nfrex:-1:1
    % create wavelet
    w          = 2*( 10/(2*pi*frex(fi)) )^2; % w is width of Gaussian
    cmwX  = fft(exp(1i*2*pi*frex(fi).*wavetime) .* exp( (-wavetime.^2)/w ), Lconv);
    cmwX  = cmwX./max(cmwX);
    
    % convolution of all trials simultaneously using bxsfun
    convres  = ifft( dataX.*cmwX );
    angleVel = diff(unwrap(angle(convres(:,halfwavsize:end-halfwavsize-1))));
    angleVel(length(t)) = angleVel(end);
    plot(t,srate*angleVel/(2*pi),'.','markersize',1)
end

%% chapter 8, exercise 3

% 3) In Chapter 3, "pre-whitening" (computing the first derivative) 
% was discussed as a strategy for making time series data more stationary, 
% and for acting as a low-pass filter. Compute instantaneous frequency on 
% six signals, before and after pre-whitening: Three shown in Figure 8.1, 
% and the same three signals but adding 20 Hz to have higher frequency signals. 
% Thus, there will be 12 signals in total. What is the effect of pre-whitening 
% on the accuracy of the estimated instantaneous frequencies, and does this 
% depend on the frequency range?

srate=1000; t=0:1/srate:5;

do20 = true;false;

figure(1+do20*1), clf

for i=1:3
    switch i
        case 1
            freqTS = linspace(1,20,length(t));
        case 2
            freqTS = abs(interp1(linspace(t(1),t(end),10),10*rand(1,10),t,'spline'));
        case 3
            freqTS = abs(mod(t,2)-1)*10;
    end
    
    centfreq = mean(freqTS) + do20*20;
    k = (centfreq/srate)*2*pi/centfreq;
    y = sin(2*pi.*centfreq.*t + k*cumsum(freqTS-mean(freqTS)));
    
    % plot time-varying frequencies
    subplot(3,2,(i-1)*2+1)
    plot(t(1:end-1),srate*diff(unwrap(angle(hilbert(y))))/(2*pi))
    set(gca,'ylim',[-1 20]+do20*20)
    
    % plot time-varying frequencies
    subplot(3,2,i*2)
    plot(t(1:end-2),srate*diff(unwrap(angle(hilbert(diff(y)))))/(2*pi))
    set(gca,'ylim',[-1 20]+do20*20)
end

%%


%% chapter 9, exercise 1

% 1) Generate a time series comprising a frequency-stationary sine wave at 10 Hz 
% and a chirp from 12 to 30 Hz, over ten seconds. In four separate simulations, 
% add noise that is variance-stationary and mean-non-stationary, variance-non-stationary 
% and mean-stationary, and so on. Perform a time-frequency analysis via Morlet wavelet 
% convolution, and plot the results. Do the different characteristics of noise 
% differentially affect the results?

srate=1000; t=0:1/srate:9;
n = length(t);

freqTS = linspace(12,30,n);
centfreq = mean(freqTS);
k = (centfreq/srate)*2*pi/centfreq;
basesig = sin(2*pi.*centfreq.*t + k*cumsum(freqTS-centfreq)) +...
    sin(2*pi*10*t);


figure(2),clf
subplot(211)
plot(t,basesig)


% filter at 11.5 and 32.5 Hz and compute instantaneous frequency again
wavetime  = -2:1/srate:2;
Lconv     = length(t)+length(wavetime)-1;
halfwavsize= floor(length(wavetime)/2);

% TF analysis with Morlet wavelet convolution
nfrex = 50;
frex  = linspace(5,40,nfrex);
ncyc = linspace(8,30,nfrex);


for simi=1:5
    
    if simi==1
        % mean and variance stationary
        signal = basesig + randn(1,n)*2;
    elseif simi==2
        % mean non-stationary, variance stationary
        signal = basesig + linspace(1,5,n) + randn(1,n)*2;
    elseif simi==3
        % mean stationary, variance non-stationary
        signal = basesig + linspace(0,5,n) .* randn(1,n)*2;
    elseif simi==4
        % mean and variance non-stationary
        signal = basesig + linspace(0,5,n) + linspace(0,5,n) .* randn(1,n)*2;
    else
        % normal signal, for comparison
        signal = basesig;
    end
    
    sigX = fft(signal,Lconv);
    tf   = zeros(length(frex),length(t));
    
    for fi=1:nfrex
        w       = 2*( ncyc(fi)/(2*pi*frex(fi)) )^2; % w is width of Gaussian
        cmwX    = fft(exp(1i*2*pi*frex(fi).*wavetime) .* exp( (-wavetime.^2)/w ), Lconv);
        cmwX    = cmwX./max(cmwX);
        convres = ifft( sigX.*cmwX );
        convres = convres(halfwavsize:end-halfwavsize-1);
        
        tf(fi,:) = abs(convres).^2;
    end
    
    if simi<5
        figure(1), subplot(2,2,simi)
    else
        figure(2), subplot(223)
    end
    
    contourf(t,frex,tf,40,'linecolor','none')
    set(gca,'clim',[0 .3])
end

%% chapter 9, exercise 2

% 2) From the code in Chapter 9.4 for simulating multivariate time series, 
% add a triangle-shaped frequency non-stationarity to one channel such that 
% there is one triangle cycle for each of the three time windows with 
% different covariances. Create a time-frequency power plot using any 
% analysis method from this book (e.g., short-time Fourier transform, complex 
% Morlet wavelet convolution, filter-Hilbert). Does the change in multivariate 
% covariance affect the time-frequency results? If so, how, and if not, why not?

srate = 1000;
t = 0:1/srate:9-1/srate;
n = length(t)/3;
v{1} = [1 .5  0; .5  1  0;  0  0  1.1];
v{2} = [1.1 .1  0; .1  1.1  0;  0  1.1 .1];
v{3} = [.8  0 .9;  0 1  0; 0 0  1.1];

% initialize matrices
mvsig = zeros(0,length(v));
for i=1:length(v)
    mvsig = cat(1,mvsig,randn(n,size(v{i},1))*chol(v{i}*v{i}'));
end

% create triangle time series
freqTS = (1.5-abs(mod(t,3)-1.5))*10;
centfreq = mean(freqTS);
k = (centfreq/srate)*2*pi/centfreq;
y = mvsig(:,1)' + sin(2*pi.*centfreq.*t + k*cumsum(freqTS-centfreq));



% filter at 11.5 and 32.5 Hz and compute instantaneous frequency again
wavetime  = -2:1/srate:2;
Lconv     = length(t)+length(wavetime)-1;
halfwavsize= floor(length(wavetime)/2);

% TF analysis with Morlet wavelet convolution
nfrex = 50;
frex  = linspace(5,20,nfrex);
ncyc  = linspace(8,20,nfrex);


sigX = fft(y,Lconv);
tf   = zeros(length(frex),length(t));

for fi=1:nfrex
    w       = 2*( ncyc(fi)/(2*pi*frex(fi)) )^2; % w is width of Gaussian
    cmwX    = fft(exp(1i*2*pi*frex(fi).*wavetime) .* exp( (-wavetime.^2)/w ), Lconv);
    cmwX    = cmwX./max(cmwX);
    convres = ifft( sigX.*cmwX );
    convres = convres(halfwavsize:end-halfwavsize-1);
    
    tf(fi,:) = abs(convres).^2;
end

clf,
contourf(t,frex,tf,40,'linecolor','none')
colormap gray
set(gca,'clim',[0 .15])
xlabel('Time (s)'), ylabel('Frequencies (Hz)')
hold on
plot([3 3],get(gca,'ylim'),'w:')
plot([6 6],get(gca,'ylim'),'w:')

%% chapter 10, exercise 2

% 2) From the three simulations above that include non-stationarities, apply pre-processing 
% strategies to minimize the non-stationarities (after the noise has already been added to 
% the sine waves signal), and then re-compute the time-frequency power. Did the pre-processing 
% successfully attenuate the noise, and did the come at the expense of reduced signal?

srate=1000; t=0:1/srate:9;
n = length(t);

freqTS = linspace(12,30,n);
centfreq = mean(freqTS);
k = (centfreq/srate)*2*pi/centfreq;
basesig = sin(2*pi.*centfreq.*t + k*cumsum(freqTS-centfreq)) +...
    sin(2*pi*10*t);

% filter at 11.5 and 32.5 Hz and compute instantaneous frequency again
wavetime  = -2:1/srate:2;
Lconv     = length(t)+length(wavetime)-1;
halfwavsize= floor(length(wavetime)/2);

% TF analysis with Morlet wavelet convolution
nfrex = 50;
frex  = linspace(5,40,nfrex);
ncyc = linspace(8,30,nfrex);


for simi=1:2
    
    if simi==1
        % mean stationary, variance non-stationary
        signal = basesig + linspace(0,5,n) .* randn(1,n)*2;
    elseif simi==2
        % mean and variance non-stationary
        signal = basesig + linspace(0,5,n) + linspace(0,5,n) .* randn(1,n)*2;
    end
    
    signal = signal./linspace(1,4,n);
    
    sigX = fft(signal,Lconv);
    tf   = zeros(length(frex),length(t));
    
    for fi=1:nfrex
        w          = 2*( ncyc(fi)/(2*pi*frex(fi)) )^2; % w is width of Gaussian
        cmwX  = fft(exp(1i*2*pi*frex(fi).*wavetime) .* exp( (-wavetime.^2)/w ), Lconv);
        cmwX  = cmwX./max(cmwX);
        convres  = ifft( sigX.*cmwX );
        convres = convres(halfwavsize:end-halfwavsize-1);
        
        tf(fi,:) = abs(convres).^2;
    end
    
%     subplot(211), plot(t,signal)
    subplot(2,2,simi+2)
    contourf(t,frex,tf,40,'linecolor','none')
    set(gca,'clim',[0 .1])
end

%% chapter 9, exercise 3

% 3) Something about FFT of stationary and non-stationary noise. Is it possible to examine FFT 
% and determine whether noise is stationary or non-stationary, and if so, might it be possible 
% to use frequency-domain attenuations to get rid of some non-stationary noise?












































%%









%%
% first, the basics:
srate     = 1000;
time      = 0:1/srate:5;
wavetime  = -2:1/srate:2;
nfrex     = 100;
frex      = logspace(log10(1),log10(40),nfrex);
n         = logspace(log10(10),log10(25),nfrex);
Lconv     = length(time)+length(wavetime)-1;
halfwavsize= floor(length(wavetime)/2);

% second, initialize matrices
sincloc  = 3;
signal   = sin(2*pi.*10*(time-sincloc))./(time-sincloc) + ...
           .6*sin(2*pi.*25*(time-.5))./(time-.5) + ...
           sin(2*pi.*15*(time-2))./(time-2) + ...
           .5*sin(2*pi.*35*(time-2.5))./(time-2.5);

       tidx = dsearchn(time',4.5);
       signal(tidx:tidx+10)=100;
       
whereNaN = find(~isfinite(signal));
signal(whereNaN) = mean(signal([whereNaN-1 whereNaN+1]));
tf       = zeros(nfrex,length(time));

% fourth, perform time-frequency decomposition
dataX = fft(signal,Lconv);

for fi=1:nfrex
    % create wavelet
    w          = 2*( n(fi)/(2*pi*frex(fi)) )^2; % w is width of Gaussian
    cmwX  = fft(exp(1i*2*pi*frex(fi).*wavetime) .* exp( (-wavetime.^2)/w ), Lconv);
    cmwX  = cmwX./max(cmwX);
    
    % convolution of all trials simultaneously using bxsfun
    convres  = ifft( dataX.*cmwX);
    tf(fi,:) = 2*abs(convres(halfwavsize:end-halfwavsize-1));
end

clf
subplot(211)
plot(time,signal)

subplot(212)
contourf(time,frex,tf,40,'linecolor','none')
set(gca,'clim',[0 5],'xlim',time([1 end]))
colormap gray

%%
