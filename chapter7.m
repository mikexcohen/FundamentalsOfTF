
% This chapter requires either the Signal Processing toolbox in Matlab,
% or the Signal package in Octave (free to install from the internet).

%% Chapter 7.1, Figure 7.1

% some random numbers and their spectrum
n  = 20;
d  = randn(n,1);
dx = fft(d);

% positive frequencies are between DC+1 and Nyquist
posF = 2:floor(n/2)+mod(n,2);
dx(posF) = dx(posF)*2; 

% negative frequencies are from Nyquist+1 to end
negF = ceil(n/2)+1+~mod(n,2):n;
dx(negF) = dx(negF)*0; % really just 0; the dx(negF)* is to make the point explicit

% and then take inverse Fourier transform
hilbertd = ifft(dx);

clf
plot(d), hold on
plot(real(hilbertd),'ro')
legend({'original';'real part of Hilbert'})

%% Chapter 7.1, Figure 7.2

% parameters for multi-frequency signal
t = 0:.01:5;
sinewave = zeros(size(t));
a = [ 10  2  5  8 ];
f = [  3  1  6 12 ];

% create signal
for i=1:length(a)
    sinewave = sinewave + a(i)*sin(2*pi*f(i)*t);
end

% take Hilbert transform. The function hilbert is in the Matlab signal
% processing toolbox or the Octave signal package. However, you can create
% your own hilbert function using the code in the previous section.
hilsine = hilbert(sinewave);

subplot(311), plot(t,sinewave)
ylabel('Amplitude')

subplot(312), plot(t,abs(hilsine).^2)
ylabel('Power')

subplot(313), plot(t,angle(hilsine))
ylabel('Phase (rad.)'), xlabel('Time (s)')

%% Chapter 7.2, Figure 7.3

t = 0:.001:1;
sinewave = 3*sin(2*pi*5*t);

% Create a dampened sine wave by scaling its Fourier coefficients by 50%.
sinewaveDamp = real(ifft( fft(sinewave)*.5) );

clf

% plot the original and the scaled signals
plot(t,sinewave), hold on
plot(t,sinewaveDamp,'r')
xlabel('Time (s)'), ylabel('Amplitude')

%% Chapter 7.2, Figure 7.4

clf

srate = 1000;
t = 0:1/srate:1;

% create signal comprising two frequencies
signal = 3*sin(2*pi*5*t) + 4*sin(2*pi*10*t);

% Compute its Fourier spectrum and convert frequencies to Hz
signalX = fft(signal)/length(t);
hz = linspace(0,srate/2,floor(length(t)/2)+1);

% find frequency indices between 8 and 14 Hz and attenuate them by 90%.
hzidx = dsearchn(hz',[8 14]');

signalXFilt = signalX; % Filt=filtered
signalXFilt(hzidx(1):hzidx(2)) = .1*signalXFilt(hzidx(1):hzidx(2));
signalFilt = real(ifft(signalXFilt)*length(t));

% plot time domain signals before and after attentuation
subplot(211)
plot(t,signal), hold on
plot(t,signalFilt,'r')
xlabel('Time (s)'), ylabel('Amplitude')
legend({'original';'attenuated'})

subplot(212)
plot(hz,2*abs(signalX(1:length(hz))),'-*'), hold on
plot(hz,2*abs(signalXFilt(1:length(hz))),'r-o')
set(gca,'xlim',[0 15])
xlabel('Frequency (Hz)'), ylabel('Amplitude')
legend({'original';'attenuated'})

%% Chapter 7.5, Figure 7.5

clf

% create a sine wave
srate = 1000;
t = 0:1/srate:5;
sinewave = sin(2*pi*2*t);

% compute its Fourier spectrum
sinewaveX  = fft(sinewave)/length(t);

% add a sudden and sharp increase in band-limited power
hz    = linspace(0,srate/2,floor(length(t)/2+1));
hzidx = dsearchn(hz',[8 9]');
sinewaveXFilt = sinewaveX;
sinewaveXFilt(hzidx(1):hzidx(2)) = .5;

% plot the original and reconstructed signals
subplot(211)
plot(t,sinewave), hold on
plot(t,real(ifft(sinewaveXFilt)*length(t)),'r')
set(gca,'ylim',[-5 5])
legend({'original';'8-9 Hz added'})

subplot(212)
plot(hz,2*abs(sinewaveX(1:length(hz))),'-*'), hold on
plot(hz,2*abs(sinewaveXFilt(1:length(hz))),'r-o')
set(gca,'xlim',[0 15])
legend({'original';'8-9 Hz added'})

% use the following line to create the figure inset.
%set(gca,'ylim',[0 .005])

%% Chapter 7.6, Figure 7.6

clf

% define signal properties
srate=1000;
t    = 0:1/srate:3;
frex = logspace(log10(.5),log10(20),10);
amps = 10*rand(size(frex));

% create signal
data = zeros(size(t));
for fi=1:length(frex)
    data=data+amps(fi)*sin(2*pi*frex(fi)*t);
end

% Fourier spectrum of data and its frequencies in Hz
dataX = fft(data);
hz = linspace(0,srate,length(t));

% define a high-pass cut-off and then filter the data
filterkernel = (1./(1+exp(-hz+7)));% 7 Hz is filter cut-off
dataXFilt = dataX;
dataXFilt = dataXFilt.*filterkernel;


subplot(211)
plot(t,data), hold on
plot(t,real(ifft(dataXFilt)),'r')
xlabel('Time (s)'), ylabel('Amplitude')
legend({'Original';'high-pass filtered'})

subplot(212)
plot(hz,2*abs(dataX/length(t)),'-o')
hold on
plot(hz,2*abs(dataXFilt)/length(t),'r-*')
set(gca,'xlim',[0 25])
xlabel('Frequency (Hz)'), ylabel('Amplitude')
legend({'Original';'high-pass filtered'})

%% Chapter 7.6, Figure 7.7

% same as above but for a low-pass filter
filterkernel = (1-1./(1+exp(-hz+7)));% 7 Hz is filter cut-off
dataXFilt = dataX;
dataXFilt = dataXFilt.*filterkernel;

clf
subplot(211)
plot(t,data), hold on
plot(t,real(ifft(dataXFilt)),'r')


subplot(212)
plot(hz,2*abs(dataX)/length(t),'-*'), hold on
plot(hz,2*abs(dataXFilt/length(t)),'r-o')
set(gca,'xlim',[0 25])

%% Chapter 7.7, Figure 7.8

% filter kernel properties
srate   = 1000;
nyquist = srate/2; 
band    = [4 8]; % Hz 
twidth  = 0.2; % width of transition zone, specified in percent around band boundaries

% define ideal shape
freqs   = [ 0 (1-twidth)*band(1) band(1) ... 
         band(2) (1+twidth)*band(2) nyquist ]/nyquist;
idealresponse = [ 0 0 1 1 0 0 ]; 

% Compute filter kernel. The function firls is in the Matlab signal
% processing toolbox, or the Octave signal package
filtO         = round(3*(srate/band(1))); 
filterweights = firls(filtO,freqs,idealresponse); 

% this next line is redundant in Matlab but necessary in Octave. The Octave
% function firls returns a row vector, but creating the next filter will
% crash if filterweights is not a column vector.
filterweights = filterweights(:)';

% Apply the kernel to the data
filtered_data = filtfilt(filterweights,1,data);

% Fourier transform of filter kernel
filterx = fft(filterweights);
% frequencies in Hz
hz = linspace(0,nyquist,floor(filtO/2)+1);

clf
subplot(211)
plot(freqs*nyquist,idealresponse), hold on
plot(hz,abs(filterx(1:length(hz))),'r-o')
set(gca,'xlim',[0 50])
legend({'ideal response';'actual response'})
xlabel('Frequency (Hz)'), ylabel('Amplitude')

subplot(212), plot(filterweights)
axis tight
xlabel('Time (ms)'), ylabel('Amplitude')

%% Chapter 7.7, Figure 7.9

% wavelet convolution
Lconv = length(filterweights)+length(data)-1;
halfwavsize = floor(length(filterweights)/2);
convres = real(ifft( fft(data,Lconv).*fft(filterweights,Lconv) ,Lconv));
convres = convres(halfwavsize:end-halfwavsize-1);

% compared with filtfilt
filtdat = filtfilt(filterweights,1,data);

clf
plot(t,data), hold on
plot(t,convres,'r')
plot(t,filtdat,'k')
xlabel('Time (s)'), ylabel('Amplitude')
legend({'original signal';'convolution';'FIR filter'})

%% Chapter 7.8, exercise 1, Figure 7.10

% First define the basics
srate=1000; nyquist=srate/2;
t=0:1/srate:4;
peakfreq = 10; % Hz

% Second, create chirp
freqTS   = linspace(5,15,length(t));
centfreq = mean(freqTS);
k        = (centfreq/srate)*2*pi/centfreq;
pow      = abs(interp1(linspace(t(1),t(end),10), ...
               10*rand(1,10),t,'spline'));
signal   = pow.*sin(2*pi.*centfreq.*t + ...
             k*cumsum(freqTS-centfreq));

% Third, wavelet convolution
wavetime = -2:1/srate:2;
Lconv = length(t)+length(wavetime)-1;
halfwavsize = floor(length(wavetime)/2);
% compute and normalize wavelet
w         = 2*( 6/(2*pi*peakfreq) )^2;
cmwX = fft(exp(1i*2*pi*peakfreq.*wavetime) .* ...
               exp( (-wavetime.^2)/w ), Lconv);
cmwX = cmwX./max(cmwX);
% run convolution
convres = ifft( fft(signal,Lconv) .* cmwX );
convres = convres(halfwavsize:end-halfwavsize-1);

% Fourth, band-pass filter
band   = [peakfreq-.5 peakfreq+.5];
twid   = 0.15;
filtO  = round(3*(srate/band(1))); 
freqs  = [ 0 (1-twid)*band(1) band(1) band(2) ...
          (1+twid)*band(2) nyquist ]/nyquist;
ires = [ 0 0 1 1 0 0 ]; 
fweights = firls(filtO,freqs,ires);
filtdat = filtfilt(fweights,1,signal);

% Fifth, plot results
clf
subplot(311)
 plot(t,signal)
subplot(312)
 plot(t,real(convres)), hold on
 plot(t,filtdat,'r')
subplot(313)
 plot(t,2*abs(convres)), hold on
 plot(t,2*abs(hilbert(filtdat)),'r')

%% end
