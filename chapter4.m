

%% Chapter 4.1, Figure 4.1

srate = 1000;
t = 0:1/srate:10;

% A complex sine wave (in the book, csw) is made by embedding 
% a sine wave in Euler's formula
complexsinewave = exp(1i*2*pi*t); % frequency implicitly is 1

% 3D plot
plot3(t,real(complexsinewave),imag(complexsinewave))

xlabel('time'), ylabel('real part'), zlabel('imaginary part')
% Activate click-and-drag; this function does not work in Octave (as of
% summer 2014). Octave activates 3D plots automatically.
rotate3d 
axis image

% Use the following line to get the second part of Figure 4.1
%view([0 90])

%% Chapter 4.2

% create a sine wave signal
signal = 2*sin(2*pi*3*t + pi/2);

% see next section for plotting

% The Fourier transform does not deal with 'time' in real units such as
% seconds; instead, 'time' is normalized from 0 to 1.
fouriertime = (0:length(t)-1)/length(t);

% initialize the Fourier coefficients vector
signalX = zeros(size(signal));

% loop through data points in the signal
% (which are converted to frequencies)
for fi=1:length(signalX)
    % complex sine wave for this frequency
    csw = exp(-1i*2*pi*(fi-1)*fouriertime);
    
    % dot product between complex sine wave and signal
    signalX(fi) = sum( csw.*signal ) / length(signal);
end

% the above loop-based algorithm for the discrete Fourier transform is slow
% and can be replaced with the Matlab function fft (fast-Fourier-transform).
signalXF = fft(signal)/length(signal);

%% 4.3, Figure 4.2

% Nyquist frequency is half of the sampling rate.
nyquistfreq = srate/2;

% Convert frequencies in arbitrary units to frequencies in Hz (1/seconds)
hz = linspace(0,nyquistfreq,floor(length(t)/2)+1);

clf
% plot the time domain signal
subplot(211)
plot(t,signal)
xlabel('Time (s)'), ylabel('Amplitude')
set(gca,'ylim',[-2.2 2.2])

% and now plot the result of the Fourier transform
subplot(212)
plot(hz,2*abs(signalX(1:length(hz))),'o'), hold on
plot(hz,2*abs(signalXF(1:length(hz))),'r.-')
set(gca,'xlim',[0 10],'ylim',[-.05 2.2])
xlabel('Frequencies (Hz)'), ylabel('Amplitude')
legend({'slow Fourier transform';'fast Fourier transform'})

%% Chapter 4.3

% optional plotting of phase values
clf
plot(hz,angle(signalX(1:length(hz))))
hold on
plot(hz,angle(signalXF(1:length(hz))),'r')
xlabel('Frequencies (Hz)'), ylabel('Phase (radians)')

%% Chapter 4.4, Figure 4.3

srate = 1000;
t = 0:1/srate:5;
a = [10 2 5 8]; 
f = [3 1 6 12];

% create multifrequency signal
signal = zeros(size(t));
for i=1:length(a)
    signal = signal + a(i)*sin(2*pi*f(i)*t);
end

% Compute its Fourier transform
signalX = fft(signal)/length(signal);
hz = linspace(0,srate/2,floor(length(t)/2)+1);

% plot time domain
subplot(211)
plot(t,signal)
xlabel('Time (s)'), ylabel('amplitude')

% plot frequency domain
subplot(212)
plot(hz,2*abs(signalX(1:length(hz))),'ks-','markerface','k')
set(gca,'xlim',[0 max(f)*1.3]);
xlabel('Frequencies (Hz)'), ylabel('amplitude')

%% Chapter 4.4, Figure 4.4

% Add noise to signal 
signalN = signal + randn(size(signal))*20;
fouriercoefsN = fft(signalN)/length(signalN);

subplot(211)
plot(t,signalN)
xlabel('Time (s)'), ylabel('amplitude')

subplot(212)
plot(hz,2*abs(fouriercoefsN(1:length(hz))),'ks-','markerface','k')
set(gca,'xlim',[0 max(f)*1.3]);
xlabel('Frequencies (Hz)'), ylabel('amplitude')

%% Chapter 4.5, Figure 4.5

% The following two lines are equivalent
[junk,tenHzidx] = min(abs(hz-10));
tenHzidx = dsearchn(hz',10);

frex_idx = sort(dsearchn(hz',f'));
requested_frequences = 2*abs(signalX(frex_idx));


clf
bar(requested_frequences)
xlabel('Frequencies (Hz)'), ylabel('Amplitude')
set(gca,'xtick',1:length(frex_idx),'xticklabel',cellstr(num2str(round(hz(frex_idx))')))

%% Chapter 4.6, Figure 4.6

srate=1000;
t = 0:1/srate:5;
f = 3; % frequency in Hz

% sine wave with time-increasing amplitude
signal = linspace(1,10,length(t)).*sin(2*pi*f*t);

signalX = fft(signal)/length(t);
hz = linspace(0,srate/2,floor(length(t)/2)+1);

subplot(211),
plot(t,signal)
xlabel('Time'), ylabel('amplitude')

subplot(212)
plot(hz,2*abs(signalX(1:length(hz))),'s-','markerface','k')
xlabel('Frequency (Hz)'), ylabel('amplitude')
set(gca,'xlim',[0 10])

%% Chapter 4.6, Figure 4.7

a = [10 2 5 8];
f = [3 1 6 12];

% This is nearly the same signal that was created in Chapter 3.
timechunks = round(linspace(1,length(t),length(a)+1));

signal = 0;
for i=1:length(a)
    signal = cat(2,signal,a(i)* ...
        sin(2*pi*f(i)*t(timechunks(i):timechunks(i+1)-1) ));
end

signalX = fft(signal)/length(t);
hz = linspace(0,srate/2,floor(length(t)/2)+1);

subplot(211)
plot(t,signal)
xlabel('Time'), ylabel('amplitude')

subplot(212)
plot(hz,2*abs(signalX(1:length(hz))),'s-','markerface','k')
xlabel('Frequency (Hz)'), ylabel('amplitude')
set(gca,'xlim',[0 20])

%% Chapter 4.6, Figure 4.8

f  = [2 10];
ff = linspace(f(1),mean(f),length(t));
signal = sin(2*pi.*ff.*t);

signalX = fft(signal)/length(t);
hz = linspace(0,srate/2,floor(length(t)/2));

subplot(211)
plot(t,signal)
xlabel('Time'), ylabel('amplitude')

subplot(212)
plot(hz,2*abs(signalX(1:length(hz))))
xlabel('Frequency (Hz)'), ylabel('amplitude')
set(gca,'xlim',[0 30])

%% 4.7, Figure 4.9

srate = 100;
t = 0:1/srate:11;
boxes = double(.02+(mod(.25+t,2)>1.5));

signalX = fft(boxes)/length(t);
hz = linspace(0,srate/2,floor(length(t)/2));

subplot(211)
plot(t,boxes)
xlabel('Time'), ylabel('amplitude')
set(gca,'xlim',[0 11],'ylim',[-.05 1.1])

subplot(212)
plot(hz,2*abs(signalX(1:length(hz))))
xlabel('Frequency (Hz)'), ylabel('amplitude')
set(gca,'xlim',[0 30],'ylim',[0 .3])

%% Chapter 4.8, Figure 4.10

x = (linspace(0,1,1000)>.5)+0; % +0 converts from boolean to number

% plot
subplot(211)
plot(x)
set(gca,'ylim',[-.1 1.1])
xlabel('Time (a.u.)'), ylabel('Amplitude (a.u.)')

subplot(212)
plot(abs(fft(x))),
set(gca,'xlim',[0 200],'ylim',[0 100])
xlabel('Frequency (a.u.)'), ylabel('Amplitude (a.u.)')

%% Chapter 4.8, Figure 4.11

srate = 1000; 
t     = 0:1/srate:10;
n     = length(t);
hann  = .5*(1-cos(2*pi*(1:n)/(n-1))); % Hann taper

% define sine and cosine waves
x1 = sin(2*pi*2*t + pi/2);
x2 = sin(2*pi*2*t);

subplot(211)
plot(t,x1)
hold on
plot(t,x2,'r')
xlabel('Time'), ylabel('Amplitude')

% FFTs
X1 = fft(x1)/length(t);
X2 = fft(x2)/length(t);
hz = linspace(0,srate/2,floor(length(t)/2)+1);

% Plot power spectra
subplot(212)
plot(hz,2*abs(X1(1:length(hz))),'b.-'), hold on
plot(hz,2*abs(X2(1:length(hz))),'r.-')
xlabel('Frequency (Hz)'), ylabel('Amplitude')
set(gca,'xlim',[0 10],'ylim',[0 .001])

%% Chapter 4.8, Figure 4.12

% Below are three different windowing functions that can be used to taper a
% signal before computing the Fourier transform. These are certainly not
% the only three, but they are the three most often used.

hann_win     = .5*(1-cos(2*pi*linspace(0,1,n)));
hamming_win  = .54 - .46*cos(2*pi*linspace(0,1,n));
gaussian_win = exp(-.5*(2.5*(-n/2:n/2-1)/(n/2)).^2);

% select which of the above to use
win2use = hamming_win;hann_win;

figure(1), clf
subplot(311), plot(t,x1)
subplot(312), plot(t,win2use)
subplot(313), plot(t,x1.*win2use)
xlabel('Time (s)')

% Plot all three for comparison. You can see that the three windows are all
% extremely similar to each other. Thus, the choice of which window to use
% is generally not a very important one.
figure(2)
plot(t,hann_win)
hold on
plot(t,hamming_win,'r')
plot(t,gaussian_win,'k')
xlabel('Time (s)'), ylabel('Amplitude (i.e., scaling')
legend({'Hann';'Hamming';'Gaussian'})

%% Chapter 4.8, Figure 4.13

clf

X1 = fft(x1)/n;
X2 = fft(x1.*hann)/n;
hz = linspace(0,srate/2,floor(length(t)/2)+1);

plot(hz,2*abs(X1(1:length(hz))),'s-','markerface','k'), hold on
plot(hz,2*abs(X2(1:length(hz))),'sr-','markerface','r')

xlabel('Frequency (Hz)'), ylabel('Amplitude')
set(gca,'xlim',[0 5],'ylim',[-.01 1.1])

%% For plot inset, use the following code

ylims = linspace(1.1,.002,200);
for yi=1:length(ylims)
    set(gca,'ylim',[0 ylims(yi)])
    pause(.01)
end

%% Chapter 4.9, Figure 4.14

clf

% time series of zeros and ones
n = 50;
x = (linspace(0,1,n)>.5)+0;

subplot(211)
plot(x)
set(gca,'ylim',[-.1 1.1])
xlabel('Time (a.u.)'), ylabel('Amplitude')

% how much zero padding to do. Try changing this between 1 and 10.
zeropadfactor = 2;

% FFTs and corresponding frequencies in Hz
X1  = fft(x,n)/n;
X2  = fft(x,zeropadfactor*n)/n;
hz1 = linspace(0,n/2,floor(n/2)+1);
hz2 = linspace(0,n/2,floor(zeropadfactor*n/2)+1);

% plot time domain signal
subplot(212)
plot(hz1,2*abs(X1(1:length(hz1))))
hold on

% ... and their power spectra
plot(hz2,2*abs(X2(1:length(hz2))),'r')
set(gca,'xlim',[0 20])
xlabel('Frequency (a.u.)'), ylabel('Amplitude')

%% Chapter 4.10, Figure 4.15

% Define "continuous" sine wave (really just a sine with relatively high
% sampling rate compared to the "measurement" sampling rates below).
srate = 1000;
t = 0:1/srate:1;
f = 30; % Hz
d = sin(2*pi*f*t);


% "Measurement" sampling rates
srates = [15 20 50 200]; % Hz

clf
for i=1:4
    subplot(2,2,i)
    
    % plot 'continuous' sine wave
    plot(t,d), hold on
    
    % plot sampled sine wave
    samples = round(1:1000/srates(i):length(t));
    plot(t(samples),d(samples),'r-','linew',2)
    
    title([ 'Sampled at ' num2str(srates(i)) ' Hz' ])
    set(gca,'ylim',[-1.1 1.1],'xtick',0:.25:1)
end

%% Chapter 4.11, Figure 4.16

% define properties of data
nTrials = 40;
srate   = 1000;
t       = 0:1/srate:5;
npnts   = length(t);
a       = [2 3 4 2];
f       = [1 3 6 12];

% create multi-frequency signal
data=zeros(size(t));
for i=1:length(a)
    data = data + a(i)*sin(2*pi*f(i)*t);
end

% add noise using bsxfun. The idea here is to add the same signal to many
% trials of noise.
dataWnoise = bsxfun(@plus,data,30*randn(nTrials,npnts));

% Hann window to taper data
hanwin = .5*(1-cos(2*pi*linspace(0,1,npnts)));

hz = linspace(0,srate/2,floor(npnts/2)+1);
x  = zeros(nTrials,length(hz));
for triali=1:nTrials
    temp = fft(hanwin.*dataWnoise(triali,:))/npnts;
    x(triali,:) = 2*abs(temp(1:length(hz)));
end

% The loop above over trials is done for illustration to make clear that
% the FFT is computed on each trial separately. However, the loop is
% unnecessary; the previous 5 lines of code are equivalent to the following
% two lines:
x = fft(bsxfun(@times,dataWnoise,hanwin),[],2)/npnts; % FFT along second dimension
x = 2*abs(x(:,1:length(hz)));

% now plot results
clf
subplot(211)
plot(t,mean(dataWnoise))
xlabel('Time (s)'), ylabel('Amplitude')

subplot(212)
plot(hz,x), hold on
plot(hz,mean(x),'k','linewidth',5)
set(gca,'xlim',[0 20])
xlabel('Frequency (Hz)'), ylabel('Amplitude')

%% Chapter 4.12, Figure 4.17

% detrend before computing SNR. It is important to detrend along the
% correct dimension. Try removing the two transpose marks below and watch
% what happens when the data are detrended over trials (which is incorrect)
% compared to over frequencies (the correct way).
detrendx = detrend(x')';
snr = mean(detrendx)./std(detrendx);
clf
plot(hz,snr)
set(gca,'xlim',[0 20])
xlabel('Frequency (Hz)'), ylabel('Signal-to-noise ratio')

%% Chapter 4.13, Figure 4.18

% This cell requires the signal processing toolbox in Matlab, for access to
% the function dpss.m As of summer 2014, it is not included in the Signal package
% in Octave. See http://wiki.octave.org/Signal_package.

srate = 1000;
t = 0:1/srate:5;
n = length(t);

% how much noise to add
noisefactor = 20;

% create multi-frequency signal
a = [2 3 4 2];
f = [1 3 6 12];
data=zeros(size(t));
for i=1:length(a)
    data = data + a(i)*sin(2*pi*f(i)*t);
end
data = data + noisefactor*randn(size(data));

% define Slepian tapers. 
tapers = dpss(n,3)';

% initialize multitaper power matrix
mtPow = zeros(floor(n/2)+1,1);
hz = linspace(0,srate/2,floor(n/2)+1);

% loop through tapers
for tapi = 1:size(tapers,1)-1 % -1 because the last taper is typically not used
    
    % scale the taper for interpretable FFT result
    temptaper = tapers(tapi,:)./max(tapers(tapi,:));
    
    % FFT of tapered data
    x = abs(fft(data.*temptaper)/n).^2;
    
    % add this spectral estimate to the total power estimate
    mtPow = mtPow + x(1:length(hz))';
end
% Because the power spectra were summed over many tapers, 
% divide by the number of tapers to get the average.
mtPow = mtPow./tapi;

% now compute the 'normal' power spectra using one taper
hann   = .5*(1-cos(2*pi*(1:n)/(n-1)));
x      = abs(fft(data.*hann)/n).^2;
regPow = x(1:length(hz)); % regPow = regular power

% Now plot both power spectra. Note that power is plotted here instead of
% amplitude because of the amount of noise. Try plotting amplitude by
% multiplying the FFT result by 2 instead of squaring.
clf
plot(hz,mtPow,'.-'), hold on
plot(hz,regPow,'r.-')
set(gca,'xlim',[0 15])

%% Chapter 4.14, Figure 4.19

% Time series of random numbers and its FFT.
xTime = randn(20,1);
xFreq = fft(xTime)/length(xTime);

% Remember that 'time' in the Fourier transform (and, thus, also in the
% inverse-Fourier transform) is normalized from 0 to almost-1.
t = (0:length(xTime)-1)'/length(xTime);

% compute inverse Fourier transform
recon_data = zeros(size(xTime));
for fi=1:length(xTime)
    % Note that a complex sine wave is created for each frequency, scaled
    % by the Fourier coefficient derived from the forward Fourier transform.
    sine_wave = xFreq(fi)*exp(1i*2*pi*(fi-1).*t);
    recon_data = recon_data + sine_wave;
end

clf
plot(xTime,'.-'), hold on
plot(real(recon_data),'ro')

% in practice it is better to use the ifft function. The result below is
% identical to the previous. The result must be scaled up because the FFT
% result above was scaled down.
recon_data = ifft(xFreq)*length(xTime);
plot(real(recon_data),'m.')

legend({'original signal';'''manual'' inverse Fourier';'ifft reconstructed'})

%% Chapter 4.15, exercise 1, Figure 4.20

% first, the basics:
srate = 1000;
time  = 0:1/srate:6;
N     = length(time);
f     = [1 5]; % in hz
ff    = linspace(f(1),mean(f),length(time));
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
xlabel('Time (s)'), ylabel('Amplitude')
legend({'original';'phase-scrambled'})

subplot(212)
hz = linspace(0,srate/2,floor(N/2)+1);
plot(hz,2*abs(dataX(1:length(hz))/N),'o'), hold on
plot(hz,2*abs(newdataX(1:length(hz))/N),'r')
set(gca,'xlim',[0 20])
xlabel('Frequency (Hz)'), ylabel('Amplitude')
legend({'original';'phase-scrambled'})

%% end
