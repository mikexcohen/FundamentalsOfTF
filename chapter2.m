%% Fundamentals of Time-Frequency Analyses in Matlab/Octave
% Mike X Cohen
% Matlab/Octave code for Chapter 2
% 
% This code accompanies the eBook, titled "Fundamentals of Time-Frequency 
% Analyses in Matlab/Octave" [Sinc(x) Press], which can be purchased through amazon
% <link here>
% Using the code without following the book may lead to confusion, incorrect data 
% analyses, and misinterpretations of results. 
% Mike X Cohen assumes no responsibility for inappropriate or incorrect use of this code.

%% Chapter 2.1

% create two variables, and multiply them together
varA = 4;
varB = 15;
varA*varB
% note that a semi-colon at the end of a line of code "suppresses" the
% output, meaning it won't appear in the Matlab command. This is different
% from having a semi-colon inside a set of numbers, which will be shown
% below.


% variables can also contain strings (letters). Variable names should be
% meaningful and interpretable.
myName = 'mike';
myAge  = 35;
% As an aside here, it is often useful to make the code look nice.
% Nice-looking code is easier to read and easier to navigate. Consider the
% visual appear of the above two lines of code compared to below.
myName='mike';
myAge=35;


% matrices can be specified using square brackets. Note the effect of the
% semi-colon in the two variables below.
a = [ 1 2 3; 4 5 6; 7 8 9 ];
b = [ 1 2 3 4 5 6 7 8 9];

%% Chapter 2.2

% Functions are useful to save you the time programming often-used
% algorithms. Below are some examples of how to use functions.

% mean (average)
a = [1 3 5 3 2 4 6 3];
meanA = mean(a);

% 10 linearly spaced numbers between 1 and 5.
a = linspace(1,5,10)

% maximum value and position (index)
a = [1 3 2];
[maxVal,maxIdx] = max(a);

% add folders to Matlab's path. Matlab will only be able to call functions
% if they in the path or in the current directory. You can permanently 
% change the path by clicking on Home -> Set Path
addpath('/path/to/file/location') % unix
addpath('C:/Users/mxc/Desktop') % windows
addpath(genpath('/path/to/file/locations')) % multiple paths

%% Chapter 2.3

% Comments (lines of text, indicated by percent sign, that are not
% evaluated by Matlab) have already been used extensively above.

% Try to build a habit of commenting your code.
a=3;
% this line is not run
% d=3;
c=4; % comments after the line can be useful
a*d % Matlab crashes here... why?
a*c

%% Chapter 2.4 

% Indexing is used to extract certain elements from a matrix.

% return the element in the 4th position
a = [1 4 2 8 4 1 9 3];
a(4)

% return elements 4-6
a([4 5 6])

% be careful of the order of the indices
a([1 3 5])
a([2 3 1])

% indexing with multiple dimensions
a = rand(4,5);
a(3,2)

% there should be one index per dimension in the matrix
a = rand(2,5,3,2,6,8,4);
a(1,1,1,1,1,1,1)

% Matrix indexing vs. linear indexing. The two indexing lines below are
% equivalent. Linear indexing (a(1,3)) is less confusing and should be
% preferred if you are new to programming.
a = [1 2 6; 3 4 9];
a(1,3)
a(5)

%% Chapter 2.5

% list all variables in the workspace
whos

% return the size of a variable
size(a)
% Note that size() returns the number of elements along all dimensions. 
% It can also be used to extract the number of elements along a specified
% dimension.
size(a,1)
size(a,2)

% A related function is length(), which returns the number of elements
% along the longest dimension.
length(a)

% remove one variable
clear a

% remove all variables that start with m
clear m*

% remove all variables from the workspace
clear

%% Chapter 2.6

% The colon is used for counting.

% integer counting
1:10

% counting with specified increases
1:2:10
1:1.94583:10

% counting backwards
10:1 % empty
10:-1:1 % works as expected

% using the colon operator for indexing
a = 1:20;
a(1:5)
a(1:3:end)

%% Chapter 2.7

% Loops and other control commands.

% this loop runs 4 times
for i=1:4
    % The function disp() prints text in the Matlab command. 
    % The function num2str() converts a number to a character.
    disp([ 'iteration ' num2str(i) ])
end

% this loop runs until the variable 'toggle' is set to true
toggle = false;
i=1;
while toggle==false
disp([ 'iteration ' num2str(i) ])
    if i==4, toggle=true; end
    % note that loops and if-statements can be condensed into one line, as
    % long as there is a comma or semi-colon separating the different parts
    % of the command.
    
    i=i+1; % The counting variable i must be explicitly incremented here, but not in the for-loop above
end

% if statements
myAge = 52;
if myAge>25 % test whether this is true
    disp('I''m older than a quarter century.')
else % if the previous statement is not true, evaluate the following
    disp('I''m younger than a quarter century.')
end


if myAge>25 && myAge<50 % combined test
    disp('I still have many years ahead of me.')
elseif myAge>49 && myAge<80
    disp('I''m a bit old.') % note that to print a single quote, you must type two single-quotes
elseif myAge>79
    disp('I''m really old.')
else
    % It is good programming practice to have a final catch statement, just
    % in case none of the previous statements are run. This can be helpful
    % when identifying errors.
    disp('My age is less than 25 or non-numeric.')
end

%% Chapter 2.8

% Scalar, vector, and matrix multiplication

% Multiplying two scalars is no problem.
a=4; b=12; % note that the two variables are defined on the same line.
a*b

% also works fine (because 'a' is a scalar -- a single number)
b=randn(4,2); % randn() generates a random numbers of size 4,2
b*a
b+a

% works fine because both matrices are the same size
a=rand(4,2); b=rand(4,2);
a.*b
a.^b
a-b

% now there is an error, because the two variables have different sizes
a=rand(4,2); b=rand(1,2);
a.*b

% repmat can be used to replicate a matrix
repmat(b,1,1) % replicate 1 time... has no effect
repmat(b,1,2)
repmat(b,2,1)
a.*repmat(b,4,1) % this works!

% bsxfun works similarly and is a bit more elegant
bsxfun(@times,a,b)
bsxfun(@minus,a,b)

%% Chapter 2.9, Figure 2.1

% basic plotting
x=1:10; y=x.^10;
plot(x,y)

% A new graph can be shown in a new figure as above, or in the current
% figure, after clearing the figure with 'clf' (clf = clear figure)
clf

% multiple lines can be plotted simultaneously
clear y
y(:,1) = x.^1.1;
y(:,2) = exp(x/10);
plot(x,y)

% alternative method for plotting multiple lines
clf
plot(x,y(:,1))
hold on
plot(x,y(:,2))
hold off

%% Chapter 2.9, Figure 2.2

% manually specifying the line colors and marker shapes and sizes
plot(x,y(:,1),'r-o','markersize',10)
hold on
plot(x,y(:,2),'m*','markersize',14)
hold off

%% Chapter 2.2, Figure 2.3

% bar plots
clf
bar(x,y(:,1))

%% Chapter 2.9

% setting axis properties
plot(x,y)
set(gca,'xlim',[0 14])

% properties are specified in pairs. Multiple pairs can be called in the
% same command
set(gca,'xlim',[.9 10.1],'ylim',[0 20])

% Rather than using gca (get current axis), axes can be assigned pointers
clf
h = axes; % create new axis, called h (h is a 'pointer' to this axis)
plot(h,x,y); % plot x,y in axis h
set(h,'ylim',[0 15])

% figure properties can be set as well
set(gcf,'number','off','name','This is a figure')

% get can be used to find values of properties
get(gca,'xlim')
get(gca,'yscale')

% list all properties
get(gca)

%% Chapter 2.10, Figure 2.4

% explicity call a new figure
figure
subplot(2,1,1), plot(rand(3))
subplot(2,1,2), plot(rand(10))

%% Chapter 2.10, Figure 2.5

% specific figures can opened.
figure(6)
% mixing different subplot organizations
subplot(2,1,1), plot(rand(3))
subplot(2,2,3), plot(rand(10))
subplot(2,2,4), plot(rand(20))

%% Chapter 2.10

% Specific figures can also be closed.
close(6)  % close figure 6 but not other figures
close     % close the most recently used figure
close all % close all figures

%% Chapter 2.11, Figure 2.6

% matrices can be mapped onto colors in an image
clf
x = [1 2; 3 4];
imagesc(x), colorbar
colormap gray

%% Chapter 2.11, Figure 2.7

% contourf is another image viewing function
w = linspace(0,1.5,300);
x = bsxfun(@times,sin(2*pi*w),sin(w)');

clf
contourf(w,w,x)

% with different parameters, contourf can make smoother plots
figure
contourf(w,w,x,40,'linecolor','none')

%% Chapter 2.11, Figure 2.8

% note the difference in the y-axis orientation between 
% imagesc and contourf!
subplot(211)
contourf(w,w,x,40,'linecolor','none')
subplot(212)
imagesc(w,w,x)

%% end
