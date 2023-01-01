clc; clear all;
% %%%%%%%%%%%%%%%%%%%%%% Evaluation for Problem 1 %%%%%%%%%%%%%%%%%%%%%%%%%
% T = zeros(80,100);
% for sim = 1:100
%     S = generate();
%     p = transmitter(S);
%     for i = 0:79
%             q = myChannel(100, 0, 0, 0, p, i);
%             f = myReceiver(q);
%             [~,t,~,~] = myDFT(f);
%             T(i+1,sim) = t;
%     end
% end
% % Convert T into time
% for i = 0:79
%     for j = 1:100
%         T(i+1,j) = (T(i+1,j)-40)*0.0625;
%     end
% end
% %Finding Mean and Standard Deviation of Time Estimate
% Tt_mean = sum(T,2)/100;
% for i = 1:80
%     std_t(i) = std(T(i,:));
% end

% %%%%%%%%%%%%%%%%%%%%%%%% Evaluation for Problem 2 %%%%%%%%%%%%%%%%%%%%%%%
% F = zeros(25,100);
% for sim = 1:100
%     S = generate();
%     p = transmitter(S);
%     for i = -1500:125:1500
%             q = myChannel(100, 0, i, 0, p, 0);
%             f = myReceiver(q);
%             [~,~,freq,~] = myDFT(f);
%             F((i+1625)/125,sim) = freq;
%     end
% end
% % Convert F into frequency
% for i = 1:25
%     for j = 1:100
%         F(i,j) = (F(i,j) - 17)*125;
%     end
% end
% %Finding Mean and Standard Deviation
% Tf_mean = sum(F,2)/100;
% for i = 1:25
%     std_f(i) = std(F(i,:));
% end

%%%%%%%%%%%%%%%% Evaluation for Problem 3 part(a), (b), (c) %%%%%%%%%%%%%%%
% FER = zeros(37,1);
% sim = 1;
% for i = -3:0.5:15
% %      while( sim<10000 && FER(2*i+7) < 50)
%      while( sim<50000)
%             % Waiting For atmost 10,000 simulations to get 50 frame errors.
%             S = generate();
%             p = transmitter(S);
%             q = myChannel(i, 0, 0, 0, p, 40); 
%             %myChannel(snr, delay_offset, freq_offset, randphase, signal, delay_sample)
%             
%             % 40 samples correspond to 0 delay. This is for part (a).
%             
%             % Replace frequency offset by 600 and delay sample by 44 for
%             % part (b).
%             
%             % Replace frequency offset by 62.5, delay offset by 8 and sample by 40 for
%             % part (c). 
% 
%             [f, max_offset] = myReceiver(q);
%             [~,time_index,freq,retr] = myDFT_rev(f);
%             Tf(2*i+7,sim) = freq;
%             Tt(2*i+7,sim) = time_index;
%             max_offset(2*i+7,sim) = max_offset;
%             BER(2*i+7,sim) = myComparator(retr,S);
%             if BER(2*i+7, sim) > 0
%                 FER(2*i+7) = FER(2*i+7) +1;
%             end
%             sim = sim+1;
%      end
%      FER(2*i+7) = FER(2*i+7)/sim;
%      sim = 1;
% end
% % Convert Tf and Tt arrays into time and frequencies
% for i = 1:37
%     for j = 1:length(Tt(i,:))
%         Tt(i,j) = (Tt(i,j)-40)*0.0625 + (0.0625/16)*(max_offset(i,j));
%     end
% end
% for i = 1:37
%     for j = 1:length(Tf(i,:))
%         Tf(i,j) = (Tf(i,j)-17)*125;
%     end
% end 
% % Finding Mean and Standard Deviation
% for i = 1:37
%     Tt_mean(i) = sum(Tt(i,:))/length(Tt(i,:)); % Mean Of Time estimate
%     Tf_mean(i) = sum(Tf(i,:))/length(Tf(i,:)); % Mean Of Frequency estimate
%     std_t(i) = std(Tt(i,:)); % Standard Deviation of Time estimate
%     std_f(i) = std(Tf(i,:)); % Standard Deviation of Frequency estimate
% end
% % Plotting FER and BER Curves
% for i = 1:37
%     SNR_array(i) = (i-7)/2.0;
%     BER_mean(i) = sum(BER(i,:))/length(BER(i,:)); % Finding mean BER over all simulations
% end
% subplot(2,2,1)
% semilogy(SNR_array, BER_mean);
% subplot(2,2,2)
% semilogy(SNR_array, FER)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FUNCTION DEFINITIONS %%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate Frame Of size 800
function [S] = generate()
    framesize = 800;  % set framesize here
    S = zeros(framesize,1);
    for i = 1:128
        S(i) = 1;
    end
    for i = 129:136
        S(i) = 0;
    end
    for i = 137:framesize
        S(i) = randi([0,1]);
    end
end

% Design Of Transmitter
function [T,T1,T2,b] = transmitter(rawData)
       framesize= 800;                
       T1 = zeros(framesize,1);
       for j = 1:framesize
           % pi/4 BPSK
           T1(j) = (1-2*rawData(j))*(cos(pi*(j-1)/4.0)+sin(pi*(j-1)/4.0)*1i); % Modulated Data
       end
       % Upsample here
       T2 = upsample(T1,16);
       % RRC Filtering
       rolloff = 0.35; % Filter rolloff
       span = 6;       % Filter span
       sps = 16;       % Samples per symbol 
       b = rcosdesign(rolloff, span, sps);
       T = conv(T2(:,1),b); 
       span = 48;
       T = T(span+1:end-span);
end

% Design Of Channel
function [R] = myChannel(snr, delay_offset, freq_offset, randphase, signal, delay_sample)
          framesize = 800;                         
          fS = 16000; % sampling frequency
          R = zeros((framesize+80)*16,1);
          delay = delay_offset+16*delay_sample;
          % delay_offset is a value between 1 and 16
          % delay_sample is a value between 0 and 79
          for j = 1:length(signal)
              R(j+delay) = signal(j)*exp(1i*(2*pi*freq_offset*(j-1)/(16*fS) + randphase));
          end
          R = awgn(R,snr,'measured');
end

% Design Of Receiver
function [F, max_offset] = myReceiver(R)
          F = zeros(880,1);
          sum = 0;
          sum_prev = 0;
          max_offset = 1;
          % RRC Filtering
          rolloff = 0.35; % Filter rolloff
          span = 6;       % Filter span
          sps = 16;       % Samples per symbol 
          b = rcosdesign(rolloff, span, sps);
          I = conv(R(:,1),b);
          span = 48;
          I = I(span+1:end-span);
          % Finding Correct Sampling Time
          for offset = 1:16
              for j = 0:39
                  sum = sum + (abs(I(16*j+offset)))^2;
              end
              if sum > sum_prev
                  sum_prev = sum;
                  max_offset = offset;
              end
              sum = 0;
          end
          % Undersampled Signal
          for j = 1:880
              F(j) = I(16*(j-1) + max_offset);
          end
end

% DFT at Receiver
function [F_dft, time_index, freq_index, retr] = myDFT(F)
    F_dft = zeros(80,1); % change this to 753 for running dft over entire frame
    for j = 1:80 % change this to 753 for running dft over entire frame. 
        F_dft(j,1) = max(fft(F(j:j+127,1)));
    end
    [~,time_index] = max(F_dft);
    [~,freq_index] = max(fft(F(time_index:time_index+127,1)));
    % Retrieved Signal
    retr = F(time_index:time_index+799,1);
    for j = 1:800
         retr(j)= retr(j)*exp(-1i*((2*pi*(freq_index-1)*(j-1)/128)));
    end
end

% DFT at Receiver revised
function [F_dft, time_index, freq_index, retr] = myDFT_rev(F)
    F_dft = zeros(80,1); % change this to 753 for running dft over entire frame
    dft_m = dftmtx(128);
    dft_m = dft_m(5:29,:); % only 25 bins are required to be computed.
    for j = 1:80 % change this to 753 for running dft over entire frame. 
        F_dft(j) = max(dft_m*F(j:j+127,1));
    end
    [~,time_index] = max(F_dft);
    [~,freq_index] = max(fft(F(time_index:time_index+127,1)));
    % Retrieved Signal
    retr = F(time_index:time_index+799,1);
    for j = 1:800
         retr(j)= retr(j)*exp(-1i*((2*pi*(freq_index-1)*(j-1)/128)));
    end
end

% BER and FER Computation
function [BER, decoded] = myComparator(retr, S)
    BER = 0;
    decoded = zeros(800,1);
    % Decode the return signal
    for i = 1:800
        if (-pi/2.0 < angle(retr(i)) && angle(retr(i)) < pi/2.0)
            decoded(i) = 0;
        else 
            decoded (i) = 1;
        end
    end
    for i = 1:800
        if decoded(i) ~= S(i)
            BER= BER+1;
        end
    end
    BER = BER/800;
end




