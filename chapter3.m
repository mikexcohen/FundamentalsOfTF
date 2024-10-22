


%% Chapter 3.1, Figure 3.1

% Call figure 1, and wipe it clean
figure(1), clf

% generate 1000 random numbers
Yu = rand(1000,1);
Yn = randn(1000,1);

% plot numbers over 'time'
subplot(211)
plot(Yu)
hold on
plot(Yn,'r')
legend({'uniform';'Gaussian'})
xlabel('''Time'''), ylabel('Value')

% plot histograms to show the distributions of uniform (rand) vs. 
% Gaussian distributed (randn) numbers.
subplot(223)
hist(Yu,200) % the ",200" means to use 200 bins in the histogram
title('Distribution of uniform noise')
xlabel('Value'), ylabel('Count')

subplot(224)
hist(Yn,200)
title('Distribution of random noise')
xlabel('Value'), ylabel('Count')

% note that the function hist() can return outputs
[y,x] = hist(Yn,200);
hold on
plot(x,y,'m') % m = magenta color

%% Chapter 3.1, Figure 3.2

% generate more noise
whitenoise = randn(10000,1);

% Compute its Fourier transform and modify to make pink noise.
% The mechanics of the next two lines will be explained in more
% detail in Chapter 5.
whitenoiseX = fft(whitenoise);
pinknoise = real(ifft( whitenoiseX .* linspace(-1,1,length(whitenoiseX))'.^2 ))*2;

clf % clear the figure
subplot(221)
plot(whitenoise), hold on
plot(pinknoise,'r')
xlabel('Time (a.u.)'), legend({'white','pink'})

subplot(222)
plot(whitenoise,pinknoise,'.')
xlabel('white noise'), ylabel('pinkified noise')

subplot(212)
plot(abs(fft(whitenoise))), hold on
plot(abs(fft(pinknoise)),'r')
legend({'white';'pink'})
set(gca,'xticklabel','') % the '' means no ticks on the x-axis
xlabel('Frequency (a.u.)'), ylabel('amplitude')

%% Chapter 3.2, Figure 3.3

clf
t = 0:.001:5; % time from 0 to 5 seconds in steps of .001
a = 10;       % amplitude
f = 3;        % frequency in Hz
p = pi/2;     % phase angle in radians

y = a*sin(2*pi*f*t+p);
plot(t,y)
xlabel('Time (s)'), ylabel('amplitude')

%% Chapter 3.2, Figure 3.4

t = 0:.001:5; % .001 means a sampling rate of 1000 Hz, or 1 kHz
a = [10 2 5 8]; % several amplitudes for several sine waves
f = [3 1 6 12];
p = [0 pi/4 -pi pi/2];

% The variable sinewave must be initialized. Why is this,
% and why would Matlab produce an error if it is not initialized?
sinewave = zeros(size(t));

% loop through and sum the sine waves
for i=1:length(a)
    sinewave = sinewave + a(i)*sin(2*pi*f(i)*t+p(i));
end

% and plot
clf
plot(t,sinewave)
xlabel('Time (s)'), ylabel('amplitude')
% note the concatenation of strings used for the title
title([ 'Sum of ' num2str(length(a)) ' sine waves' ])

%% Chapter 3.2 

% adding random noise
sinewave = sinewave + mean(a)*randn(size(t));
plot(t,sinewave)
xlabel('Time (s)'), ylabel('amplitude')
title('With noise added...')

%% Chapter 3.2, Figure 3.5

% specify amplitudes and frequencies
a = [10  2  5  8]; % arbitrary units
f = [ 3  1  6 12]; % Hz

% Define time windows for the different sine waves
timechunks = round(linspace(1,length(t),length(a)+1));

% This time, sinewave is initialized to one zero, because it will be
% concatenated with other sine waves.
sinewave = 0;
for i=1:length(a)
    % three periods (an elipsis) can be used to continue long lines of code
    % to the next line. They are never necessary but can be useful,
    % particularly for people who have an aversion to the horizontal scroll bar.
    sinewave = cat(2,sinewave,a(i)* ...
                sin(2*pi*f(i)*t(timechunks(i):timechunks(i+1)-1) ));
end

plot(t,sinewave)
% You can copy the x/y-axis labels from above or create your own.

%% Chapter 3.2, Figure 3.6

% Create a chirp. A "chirp" is a sine wave with a frequency that varies
% over time. Chapter 8 will show additional ways to produce time series
% with arbitrary time-varying frequencies.
f = [2 10];
ff = linspace(f(1),mean(f),length(t));
sinewave = sin(2*pi.*ff.*t);
plot(t,sinewave)
set(gca,'ylim',[-1.4 1.4])

%% Chapter 3.2, Figure 3.7

% This sine wave has a constant frequency and 
% amplitude that varies over time
a = linspace(1,10,length(t)); % time-varying amplitude
f = 3; % frequency in Hz

y = a.*sin(2*pi*f*t);
plot(t,y)

%% Chapter 3.3, Figure 3.8

t = -1:.001:5;
s = [ .5 .1 ];
a = [  4  5 ];

g1 = a(1)*exp( (-t.^2) /(2*s(1)^2) );
g2 = a(2)*exp( (-(t-mean(t)).^2) /(2*s(2)^2) );

clf
plot(t,g1), hold on
plot(t,g2,'r')
legend({'Gaussian 1';'Gaussian 2'})
xlabel('Time (s)'), ylabel('Amplitude')
title('Two Gaussians with different properties')

%% Chapter 3.4, Figure 3.9

clf
t = 0:.01:11;
plot(t,mod(t,2)>1), hold on
plot(t,.02+(mod(.25+t,2)>1.5),'r')
set(gca,'ylim',[-.1 1.1],'xlim',[0 11])

%% Chapter 3.4, Figure 3.10

clf
t = 0:.01:11;
plot(t,abs(mod(t,2)-1.25)), hold on
plot(t,abs(mod(t,2)-1),'r')
set(gca,'ylim',[-.05 1.3],'xlim',[0 11])

%% Chapter 3.5, Figure 3.11

% some interesting time series...

srate = 100; % Hz
t = 1:1/srate:11; % note that time is now specified as 1/srate
subplot(221)
plot(t,sin(exp(t-5)))
set(gca,'xlim',[1 11])

subplot(222)
plot(t,log(t)./t.^2)
set(gca,'xlim',[1 11])

subplot(223)
plot(t,sin(t).*exp((-(t-3).^2)))
set(gca,'xlim',[1 11])

subplot(224)
plot(t,abs(mod(t,2)-.66).*sin(2*pi*10*t))
set(gca,'xlim',[1 11])

% The above four time series were created by combining different parts of
% time series shown earlier. Try making up some new time series on your
% own. Perhaps you can make even more interesting plots than the ones shown
% here.

%% Chapter 3.6, Figure 3.12

clf

% 51 time points of arbitrary units.
t = 0:50;

% initialize random numbers...
a = randn(size(t));
% ... and then add to each number weighted previous numbers.
a(3:end) = a(3:end) + .2*a(2:end-1) - .4*a(1:end-2);

% An alternative approach to the above, and one that produces
% non-stationary time series.
b1=rand;
for i=2:length(a)
    b1(i) = 1.05*b1(i-1) + randn/3;
end

subplot(221), plot(t,a)
title('Stationary process')

subplot(222), plot(t,b1)
title('Non-stationary process')

% plot the detrended data, which can sometimes remove non-stationarities
subplot(223), plot(t,detrend(b1))
title('detrended non-stationary process')


% computing the derivative can remove some non-stationarities
b1deriv = diff(b1); 
% Because the derivative makes the time series one point smaller,
% it is sometimes useful to add an extra point to the end ("padding")
% to facilitate plotting and analyses.
b1deriv(end+1) = b1deriv(end);

subplot(224), plot(t,b1deriv)
title('Derivative of non-stationary process')

%% Chapter 3.7, Figure 3.13

clf
% create two time series of the same frequency 
% but differing sampling rates
srates = [100 1000];
t1 = 0:1/srates(1):2; % time vector 1
t2 = 0:1/srates(2):2; % time vector 2

% now the two sine waves. Note that they are identical
% except for the time vector (i.e., sampling rate).
% Also notice that no frequency is specified, which means
% it is implicitly set to 1.
sine1 = sin(2*pi*t1);
sine2 = sin(2*pi*t2);

% plot the lines
plot(t1,sine1,'bo','markersize',9)
hold on
plot(t2,sine2,'r.','markersize',6)

%% Chapter 3.7, Figure 3.13

% same as above but for different sampling rates

clf
srates = [100 3];
t1 = 0:1/srates(1):2;
t2 = 0:1/srates(2):2;

sine1 = sin(2*pi*t1);
sine2 = sin(2*pi*t2);


plot(t1,sine1,'bo','markersize',9), hold on
plot(t2,sine2,'r.-','markersize',29)

% try changing the sampling rate to other values to see what the effects
% are on the ability to measure the 1-Hz sine wave.

%% Chapter 3.8, Figure 3.15

clf

% create time series
srate = 100;
t = 0:1/srate:10-(1/srate);
x = sin(t).*exp((-(t-3).^2));

% reflection
xflip = x(end:-1:1);
reflectedX = [xflip x xflip];

% plotting
subplot(211)
plot(t,x)
set(gca,'xlim',[-10 20])
title('Original time series')

subplot(212)
plot(reflectedX)
set(gca,'xtick',[1000 1500 2000],'xticklabel',round(t([1 500 end])))
title('Time series with reflection')

%% Chapter 3.9

% covariance matrix
v = [1 .5 0; .15 1 0; 0 0 1];
c = chol(v*v');

% n time points
n = 10000;
d = randn(n,size(v,1))*c;

% highly correlated...
corr(reshape(v*v',[],1),reshape(cov(d),[],1))

clf
plot(1:n,d)
xlabel('Time (a.u.)'), ylabel('Values')

%% Chapter 3.10, Figure 3.17

%%% Solution to exercise 1
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

%% end
