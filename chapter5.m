
%% Chapter 5.1, Figure 5.1 (top panel)

% time and frequency parameters
srate = 1000;
t = 0:1/srate:5;
f = [30 3 6 12];

% define 'chunks' of time for different frequency components
timechunks = round(linspace(1,length(t),length(f)+1));

% create signal as concatenated sine waves
data = 0;
for i=1:length(f)
    data = cat(2,data,sin(2*pi*f(i)*t(timechunks(i):timechunks(i+1)-1) ));
end

figure(1), clf
subplot(211)
plot(t,data)
xlabel('Time (s)'), ylabel('Amplitude')

%% Chapter 5.1, Figure 5.1 (bottom panel)

% short-time FFT parameters
fftWidth_ms = 1000;
fftWidth    = round(fftWidth_ms/(1000/srate)/2);
Ntimesteps  = 10; % number of time widths
centertimes = round(linspace(fftWidth+1,length(t)-fftWidth,Ntimesteps));

% frequencies in Hz
hz = linspace(0,srate/2,fftWidth-1);

% initialize matrix to store time-frequency results
tf = zeros(length(hz),length(centertimes));

% Hann window to taper the data in each window for the FFT
hanwin = .5*(1-cos(2*pi*(1:fftWidth*2)/(fftWidth*2-1)));

% loop through center time points and compute Fourier transform
for ti=1:length(centertimes)
    % get data from this time window
    temp = data(centertimes(ti)-fftWidth:centertimes(ti)+fftWidth-1);
    % Fourier transform
    x = fft(hanwin.*temp)/fftWidth*2;
    
    % enter amplitude into tf matrix
    tf(:,ti) = 2*abs(x(1:length(hz)));
end

% plot the time-frequency result
subplot(212)
contourf(t(centertimes),hz,tf,1)
set(gca,'ylim',[0 40],'clim',[0 1],'xlim',[0 5])
xlabel('Time (s)'), ylabel('Frequency (Hz)')

% The figures in the book use a reverse colorscaling, 
% which can be obtained using the following code.
c = gray; % 'gray' is the colormap
colormap(c(end:-1:1,:)) % reversed colormap

%% Chapter 5.1, Figure 5.2

freq2plot = dsearchn(hz',6);

clf
plot(t(centertimes),tf(freq2plot,:),'-o')
set(gca,'xlim',[0 5],'ylim',[-.05 2.05])
xlabel('Time (s)'), ylabel('Amplitude')

%% Chapter 5.4

% Rather than storing all possible frequencies, this analysis will 
% keep only 20 evenly spaced frequencies between 0 and Nyquist.
freqs2keep = linspace(0,hz(end),30);
freqsidx   = dsearchn(hz',freqs2keep');

% initialzie new TF matrix
tf = zeros(length(freqs2keep),length(centertimes));

% and perform the short-time Fourier analysis similar to previously.
for ti=1:length(centertimes)
    x = fft(hanwin.*data(centertimes(ti)-fftWidth...
        :centertimes(ti)+fftWidth-1))/fftWidth*2;
    tf(:,ti) = 2*abs(x(freqsidx));
end

figure(2)
subplot(212)
contourf(t(centertimes),freqs2keep,tf,1)
set(gca,'ylim',[0 40],'clim',[0 1],'xlim',[0 5])
xlabel('Time (s)'), ylabel('Frequency (Hz)')

colormap(c(end:-1:1,:)) % reversed colormap

% The resulting TF map does not accurately portray the frequency content of
% the signal. Why is this? Is it related to the sub-sampling of
% frequencies? Try running the above code again, but use 20 frequencies
% between 0 and 40 Hz, rather than between 0 and the Nyquist. What does
% this tell you about the importance of selecting the appropriate frequency
% range and sampling when extracting only selective frequencies from the
% Fourier transform?

%% Chapter 5.5

% the window of time used to compute the FFT changes as a function of
% frequency. This adjusts the trade-off between temporal precision and
% frequency resolution.

% Specify the number of time window widths
NtimeWidths = 5;

% window widths will vary from 1300 to 500 ms
fftWidth_ms = linspace(1300,500,NtimeWidths);
% after specifying the widths in ms, convert to time points
fftWidth    = round(fftWidth_ms./(1000/srate)/2);

% still 10 time steps
Ntimesteps  = 10; % number of time widths
centertimes = round(linspace(max(fftWidth)+1,length(t)-max(fftWidth),Ntimesteps));

% specify frequencies to extract
freqs2keep = linspace(1,50,40);
% the number of frequencies per window width bin
freqsPerBin = ceil((freqs2keep./max(freqs2keep))*NtimeWidths);

% initialize time-frequency output matrix
tf=zeros(length(freqs2keep),length(centertimes));

% analysis...
for ti=1:length(centertimes)
    
    % loop through the number of window widths
    for fi=1:NtimeWidths
        
        % because the amount of time changes, 
        % so does the frequency resolution.
        hz = linspace(0,srate/2,fftWidth(fi)-1);
        
        % match the 'requested' frequencies to the actual frequencies
        freqsidx = dsearchn(hz',freqs2keep(freqsPerBin==fi)');
        
        % also the Hann taper must be modified for each time window
        hanwin = .5*(1-cos(2*pi*(1:fftWidth(fi)*2)/(fftWidth(fi)*2-1)));
        
        % finally, extract the time window and compute the FFT
        tempdata = data(centertimes(ti)-fftWidth(fi):centertimes(ti)+fftWidth(fi)-1);
        x = fft(hanwin.*tempdata)/fftWidth(fi)*2;
        
        % and enter into TF matrix
        tf(freqsPerBin==fi,ti) = 2*abs(x(freqsidx));
    end
end

clf

subplot(212)
contourf(t(centertimes),freqs2keep,tf,1)
set(gca,'ylim',[0 40],'clim',[0 1],'xlim',[0 5])
xlabel('Time (s)'), ylabel('Frequency (Hz)')

colormap(c(end:-1:1,:)) % reversed colormap

%% Chapter 5.7, Figure 5.3

% clear workspace
clear

% intialize...
srate = 1000;
t     = 0:1/srate:6;
N     = length(t);

% create double-chirp with no noise
f    = [2 10];
ff   = linspace(f(1),mean(f),N);
sig1 = sin(2*pi.*ff.*t);

% create a different double-chirp with noise
f    = [20 8];
ff   = linspace(f(1),mean(f),N);
sig2 = sin(2*pi.*ff.*t);

% create signals 1 and 2
data{1} = sig1 + sig2;
data{2} = sig1 + sig2 + randn(1,N)*3;

% create signal 3 from two sincs
% sinc 1
f=10; m=1;
sig1 = sin(2*pi*f*(t-m))./(t-m);

% sinc 2
f=15; m=5;
sig2 = sin(2*pi*f*(t-m))./(t-m);

% sum two sincs to get the third signal
data{3} = sig1 + sig2;

% Sincs contain NaN's, which should be removed prior to analyses. 
% They are here removed by averaging together neighboring time points.
% In theory, it is also possible to create the sinc adding eps to the
% denominator; however, that produces a sharp edge at the center of the
% sinc function and is thus not recommended.
wherenan = find(~isfinite(data{3}));
for i=1:length(wherenan)
    data{3}(wherenan(i)) = mean(data{3}([wherenan(i)-1 wherenan(i)+1]));
end


% analyze and plot
clf
for datai=1:3
    
    %% plot time series data
    
    subplot(3,3,datai)
    plot(t,data{datai})
    
    %% Wigner-Ville
    
    % initialize output matrix
    wig = zeros(numel(data{datai}));
    
    % loop through time points
    for ti=1:N
        
        % To compute the Wigner distribution, an increasing number of time
        % points is extracted, which wrap around the signal.
        tmax    = min([ti-1,N-ti,round(N/2)-1]);
        tpnts   = -tmax:tmax;
        indices = rem(N+tpnts,N)+1;
        
        % Next, the data from this time point forwards are multiplied by
        % this time point backwards.
        wig(indices,ti) = data{datai}(ti+tpnts) .* data{datai}(ti-tpnts);
    end
    
    % After the distribution is built, take the 2-D FFT
    wig   = 2*abs(fft(wig)/size(wig,1));
    hzWig = linspace(0,srate/2,N);

    % and plot the results
    subplot(3,3,datai+3)
    imagesc(t,hzWig,wig)
    axis xy, set(gca,'ylim',[0 20])
    
    %% short-time FFT, for comparison
    
    % this is a normal short-time FFT as computed earlier in this chapter.
    fftWidth_ms = 1000;
    fftWidth    = round(fftWidth_ms/(1000/srate)/2);
    Ntimesteps  = 300; % number of time widths
    centertimes = round(linspace(fftWidth+1,length(t)-fftWidth,Ntimesteps));
    
    hzST   = linspace(0,srate/2,fftWidth-1);
    tf     = zeros(length(hzST),length(centertimes));
    hanwin = .5*(1-cos(2*pi*(1:fftWidth*2)/(fftWidth*2-1)));
    
    for ti=1:length(centertimes)
        temp = data{datai}(centertimes(ti)-fftWidth:centertimes(ti)+fftWidth-1);
        x = fft(hanwin.*temp)/fftWidth*2;
        tf(:,ti) = 2*abs(x(1:length(hzST)));
    end
    
    subplot(3,3,datai+6)
    contourf(t(centertimes),hzST,tf,40,'linecolor','none')
    set(gca,'ylim',[0 20],'xlim',t([1 end]))
    
end

%% Chapter 5.8, exercise 1, Figure 5.4

clf

% first, create chirp
srate = 100;
t     = 0:1/srate:5;
f     = [1 40];
ff    = linspace(f(1),mean(f),length(t));
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

c=gray; colormap(c(end:-1:1,:));

%% end.
