%% Part1: BPSK, QPSK, QAM, 8PSK
%Initialization of variables
A=1; %trasmitted voltage level
num_bits = 1200000; %number of bits in stream
data_bits = randi([0,1], 1, num_bits); %generating a stream of random bits
SNR_range = -2 : 5; %SNR range

%-----------------------------1.1 Mapper------------------------------------
%Mapping BPSK and getting its symbols
BPSK_symbols = (2*data_bits - 1) * A; %Transmitting 1 as 1 V and 0 as -1 V

%Mapping QPSK and getting its symbols
%QPSK_symbols_1 is mapped using grey code
QPSK_symbols_1 = zeros(1,ceil(num_bits/2));
%QPSK_symbols_2 is mapped without using grey code
QPSK_symbols_2 = zeros(1,ceil(num_bits/2));
for k=1:2:num_bits     
    if data_bits(k) && data_bits(k+1)
        %the data is stored once each two bits, so the formula of the
        %columns is changed to store one value for each two consecutive
        %bits and get a new array of symbols have the size of half the
        %number of bits correclty
        QPSK_symbols_1(1,k-floor(k/2))= 1+1i;
        QPSK_symbols_2(1,k-floor(k/2))= 1-1i;
    elseif data_bits(k) && ~data_bits(k+1)
        QPSK_symbols_1(1,k-floor(k/2))= 1-1i;
        QPSK_symbols_2(1,k-floor(k/2))= 1+1i;
    elseif ~data_bits(k) && ~data_bits(k+1)
        QPSK_symbols_1(1,k-floor(k/2))= -1-1i;
        QPSK_symbols_2(1,k-floor(k/2))= -1-1i;
    else
        QPSK_symbols_1(1,k-floor(k/2))= -1+1i;
        QPSK_symbols_2(1,k-floor(k/2))= -1+1i;
    end
end

%Mapping QAM and getting its symbols
QAM_symbols = zeros(1,ceil(num_bits/4));
for k=1:4:num_bits
    if data_bits(k) && data_bits(k+1)
        %the data is stored once each four bits, so the formula of the
        %columns is changed to store one value for each four consecutive
        %bits and get a new array of symbols have the size of one fourth the
        %number of bits correclty
        QAM_symbols(1,k-(floor(k/4)*3))= 1;
    elseif data_bits(k) && (~data_bits(k+1))
        QAM_symbols(1,k-(floor(k/4)*3))= 3;
    elseif (~data_bits(k)) && data_bits(k+1)
        QAM_symbols(1,k-(floor(k/4)*3))= -1;
    else 
        QAM_symbols(1,k-(floor(k/4)*3))= -3;
    end
    if data_bits(k+2) && data_bits(k+3)
        QAM_symbols(1,k-(floor(k/4)*3))=QAM_symbols(1,k-(floor(k/4)*3))+ 1i;
    elseif data_bits(k+2) && (~data_bits(k+3))
        QAM_symbols(1,k-(floor(k/4)*3))=QAM_symbols(1,k-(floor(k/4)*3))+ 3i;
    elseif (~data_bits(k+2)) && data_bits(k+3)
        QAM_symbols(1,k-(floor(k/4)*3))=QAM_symbols(1,k-(floor(k/4)*3))- 1i;
    else 
        QAM_symbols(1,k-(floor(k/4)*3))=QAM_symbols(1,k-(floor(k/4)*3))- 3i;
    end        
end

%Mapping 8PSK and getting its symbols
PSK8_symbols = zeros(1,ceil(num_bits/3));
for k=1:3:num_bits
    one_symbol= (data_bits(k)*100) + (data_bits(k+1)*10) + data_bits(k+2);
    switch one_symbol
        case 000  
            %the data is stored once each three bits, so the formula of the
            %columns is changed to store one value for each three consecutive
            %bits and get a new array of symbols have the size of one third 
            %the number of bits correclty
            PSK8_symbols(1,k-(floor(k/3)*2)) = 1;
        case 001  
            PSK8_symbols(1,k-(floor(k/3)*2)) = (1/sqrt(2)) + (1i/sqrt(2));
        case 011
            PSK8_symbols(1,k-(floor(k/3)*2)) = 1i;
        case 010
            PSK8_symbols(1,k-(floor(k/3)*2)) = -(1/sqrt(2)) + (1i/sqrt(2));
        case 110
            PSK8_symbols(1,k-(floor(k/3)*2)) = -1;
        case 111
            PSK8_symbols(1,k-(floor(k/3)*2)) = -(1/sqrt(2)) - (1i/sqrt(2));
        case 101
            PSK8_symbols(1,k-(floor(k/3)*2)) = -1i;
        case 100
            PSK8_symbols(1,k-(floor(k/3)*2)) = (1/sqrt(2)) - (1i/sqrt(2));
    end
end

%------------------1.2 Adding noise due to channel---------------------------
%generating normally distributed noise with zero mean and variance=1
BPSK_noise = randn(1, num_bits);

%generating normally distributed noise with zero mean and variance=2
%due to adding two randn functions 
QPSK_noise = randn(1, num_bits/2) + (randn(1, num_bits/2))*1i;
QAM_noise = randn(1, num_bits/4) + (randn(1, num_bits/4))*1i;
PSK8_noise = randn(1, num_bits/3) + (randn(1, num_bits/3))*1i;

Eb = 1;         % to have a symbol energy equal to one for BPSK and QPSK
Eb_QAM = 2.5;   % to have a symbol energy equal to one for the QAM
Eb_8PSK = 1/3;  % to have a symbol energy equal to one for the 8PSK

No = zeros (1,8); %initialization of No used in the variance of noise (No/2)
                  %for BPSK and QPSK
No_QAM = zeros(1,8);%initialization of No used in the variance of noise (No/2) 
                    %for QAM
No_8PSK = zeros(1,8);%initialization of No used in the variance of noise (No/2) 
                     %for 8PSK

%initializing empty variables to store the scaled noise
scaled_noise_BPSK = zeros (8,num_bits);
scaled_noise_QPSK = zeros (8,num_bits/2);
scaled_noise_QAM = zeros (8,num_bits/4);
scaled_noise_8PSK = zeros (8,num_bits/3);

%initalizing empty variables for demapped signal
demapped_BPSK = zeros(8,num_bits); 
demapped_QPSK_1 = zeros(8,num_bits);
demapped_QPSK_2 = zeros(8,num_bits);
demapped_QAM  = zeros(8,num_bits);
demapped_8PSK = zeros(8,num_bits);

%initializing empty variables for error
error_BPSK = zeros(8,num_bits); 
error_QPSK_1 = zeros(8,num_bits);
error_QPSK_2 = zeros(8,num_bits);
error_QAM  = zeros(8,num_bits); 
error_8PSK = zeros(8,num_bits); 

%initializing empty variable for bit error rate
BER_BPSK   = zeros(1,8);
BER_QPSK_1   = zeros(1,8);
BER_QPSK_2   = zeros(1,8);
BER_QAM  = zeros(1,8);
BER_8PSK   = zeros(1,8);

%for loop that gets the required BER from the range of SNR [-2,5] db after
%adding the noise to the signal and demapping the received signal
for SNR = SNR_range
    %Calculating No for BPSK, QPSK, QAM and 8PSK
    No(SNR+3) = Eb/(10^(SNR/10)); %BPSK and QPSK No
    No_QAM(SNR+3) = Eb_QAM/(10^(SNR/10)); %QAM No
    No_8PSK(SNR+3) = Eb_8PSK/(10^(SNR/10)); %8PSK No

    %making the noise with variance No/2 by multipying it by sqrt(No/2)
    scaled_noise_BPSK(SNR+3,:) = BPSK_noise*sqrt(No(SNR+3)/2);
    scaled_noise_QPSK(SNR+3,:) = QPSK_noise*sqrt(No(SNR+3)/2);
    scaled_noise_QAM(SNR+3,:) = QAM_noise*sqrt(No_QAM(SNR+3)/2);
    scaled_noise_8PSK(SNR+3,:) = PSK8_noise*sqrt(No_8PSK(SNR+3)/2);
    
    %getting the transmitted signal with noise added to it
    v_BPSK = BPSK_symbols + scaled_noise_BPSK(SNR+3,:);
    v_QPSK_1 = QPSK_symbols_1 + scaled_noise_QPSK(SNR+3,:);
    v_QPSK_2 = QPSK_symbols_2 + scaled_noise_QPSK(SNR+3,:);
    v_QAM = QAM_symbols + scaled_noise_QAM(SNR+3,:);
    v_8PSK = PSK8_symbols + scaled_noise_8PSK(SNR+3,:);
    
   %------------------------1.3 Start demapping----------------------------
    for k=1:num_bits
        %demapping BPSK
        if v_BPSK(k) > 0
            demapped_BPSK(SNR+3,k) = 1;
        else
            demapped_BPSK(SNR+3,k) = 0;
        end
        
        %demapping QPSK
        if mod(k,2)
            if real(v_QPSK_1(k-floor(k/2))) > 0
                demapped_QPSK_1(SNR+3,k) = 1;
            else
                demapped_QPSK_1(SNR+3,k) = 0;
            end
            if imag(v_QPSK_1(k-floor(k/2))) > 0
                demapped_QPSK_1(SNR+3,k+1) = 1;
            else
                demapped_QPSK_1(SNR+3,k+1) = 0;
            end
            if imag(v_QPSK_2(k-floor(k/2))) > 0 && ...
                    real(v_QPSK_2(k-floor(k/2))) > 0
                demapped_QPSK_2(SNR+3,k) = 1;
                demapped_QPSK_2(SNR+3,k+1) = 0;
            elseif imag(v_QPSK_2(k-floor(k/2))) > 0 && ...
                    real(v_QPSK_2(k-floor(k/2))) < 0
                demapped_QPSK_2(SNR+3,k) = 0;
                demapped_QPSK_2(SNR+3,k+1) = 1;
            elseif imag(v_QPSK_2(k-floor(k/2))) < 0 && ...
                    real(v_QPSK_2(k-floor(k/2))) < 0
                demapped_QPSK_2(SNR+3,k) = 0;
                demapped_QPSK_2(SNR+3,k+1) = 0;
            else
                demapped_QPSK_2(SNR+3,k) = 1;
                demapped_QPSK_2(SNR+3,k+1) = 1;
            end
        end
        
        %demapping QAM
        if mod(k,4)==1
            if real(v_QAM(k-floor(k/4)*3)) > 2
                demapped_QAM(SNR+3,k) = 1;
                demapped_QAM(SNR+3,k+1) = 0;
            elseif real(v_QAM(k-floor(k/4)*3)) > 0
                demapped_QAM(SNR+3,k) = 1;
                demapped_QAM(SNR+3,k+1) = 1;
            elseif real(v_QAM(k-floor(k/4)*3)) > -2
                demapped_QAM(SNR+3,k) = 0;
                demapped_QAM(SNR+3,k+1) = 1;
            else 
                demapped_QAM(SNR+3,k) = 0;
                demapped_QAM(SNR+3,k+1) = 0; 
            end
            
            if imag(v_QAM(k-floor(k/4)*3)) > 2
                demapped_QAM(SNR+3,k+2) = 1;
                demapped_QAM(SNR+3,k+3) = 0;
            elseif imag(v_QAM(k-floor(k/4)*3)) > 0
                demapped_QAM(SNR+3,k+2) = 1;
                demapped_QAM(SNR+3,k+3) = 1;
            elseif imag(v_QAM(k-floor(k/4)*3)) > -2
                demapped_QAM(SNR+3,k+2) = 0;
                demapped_QAM(SNR+3,k+3) = 1;
            else 
                demapped_QAM(SNR+3,k+2) = 0;
                demapped_QAM(SNR+3,k+3) = 0; 
            end
            
        end
        
        %demapping 8PSK
        if mod(k,3)==1
            if (-pi/8) <= angle(v_8PSK(k-floor(k/3)*2)) && ...
                    angle(v_8PSK(k-floor(k/3)*2)) < (pi/8)
                 demapped_8PSK(SNR+3,k)  = 0;
                 demapped_8PSK(SNR+3,k+1)= 0;
                 demapped_8PSK(SNR+3,k+2)= 0;
            elseif (pi/8) <= angle(v_8PSK(k-floor(k/3)*2)) && ...
                    angle(v_8PSK(k-floor(k/3)*2)) < (3*pi/8)
                 demapped_8PSK(SNR+3,k)  = 0;
                 demapped_8PSK(SNR+3,k+1)= 0;
                 demapped_8PSK(SNR+3,k+2)= 1;
            elseif (3*pi/8) <= angle(v_8PSK(k-floor(k/3)*2)) && ...
                    angle(v_8PSK(k-floor(k/3)*2)) < (5*pi/8)
                 demapped_8PSK(SNR+3,k)  = 0;
                 demapped_8PSK(SNR+3,k+1)= 1;
                 demapped_8PSK(SNR+3,k+2)= 1;
            elseif (5*pi/8) <= angle(v_8PSK(k-floor(k/3)*2)) && ...
                    angle(v_8PSK(k-floor(k/3)*2)) < (7*pi/8)
                 demapped_8PSK(SNR+3,k)  = 0;
                 demapped_8PSK(SNR+3,k+1)= 1;
                 demapped_8PSK(SNR+3,k+2)= 0;
            elseif (-7*pi/8) <= angle(v_8PSK(k-floor(k/3)*2)) && ...
                    angle(v_8PSK(k-floor(k/3)*2)) < (-5*pi/8)
                 demapped_8PSK(SNR+3,k)  = 1;
                 demapped_8PSK(SNR+3,k+1)= 1;
                 demapped_8PSK(SNR+3,k+2)= 1;
            elseif (-5*pi/8) <= angle(v_8PSK(k-floor(k/3)*2)) && ...
                    angle(v_8PSK(k-floor(k/3)*2)) < (-3*pi/8)
                 demapped_8PSK(SNR+3,k)  = 1;
                 demapped_8PSK(SNR+3,k+1)= 0;
                 demapped_8PSK(SNR+3,k+2)= 1;
            elseif (-3*pi/8) <= angle(v_8PSK(k-floor(k/3)*2)) && ...
                    angle(v_8PSK(k-floor(k/3)*2)) < (-pi/8)
                 demapped_8PSK(SNR+3,k)  = 1;
                 demapped_8PSK(SNR+3,k+1)= 0;
                 demapped_8PSK(SNR+3,k+2)= 0;
            else
                 demapped_8PSK(SNR+3,k)  = 1;
                 demapped_8PSK(SNR+3,k+1)= 1;
                 demapped_8PSK(SNR+3,k+2)= 0;
            end
        end
    end
    

    %-----------------1.4 Bit error rate calculation------------------------
    %recognize the bits that differed between transmitted and received streams 
    error_BPSK(SNR+3,:) = demapped_BPSK(SNR+3,:) == data_bits;
    error_QPSK_1(SNR+3,:) = demapped_QPSK_1(SNR+3,:) == data_bits;
    error_QPSK_2(SNR+3,:) = demapped_QPSK_2(SNR+3,:) == data_bits;
    error_QAM(SNR+3,:)  = demapped_QAM(SNR+3,:)  == data_bits;
    error_8PSK(SNR+3,:) = demapped_8PSK(SNR+3,:) == data_bits;
    
    %counting the number of flipped bits and get the ratio
    %between num of bits that was flipped and the total number of bits in
    %each stream for each SNR to get bit error rate for each SNR value
    BER_BPSK(SNR+3) = num_bits - nnz(error_BPSK(SNR+3,:));
    BER_BPSK(SNR+3) = BER_BPSK(SNR+3)/num_bits;
    
    BER_QPSK_1(SNR+3) = num_bits - nnz(error_QPSK_1(SNR+3,:));
    BER_QPSK_1(SNR+3) = BER_QPSK_1(SNR+3)/num_bits;
    
    BER_QPSK_2(SNR+3) = num_bits - nnz(error_QPSK_2(SNR+3,:));
    BER_QPSK_2(SNR+3) = BER_QPSK_2(SNR+3)/num_bits;
    
    BER_QAM(SNR+3)  = num_bits - nnz(error_QAM(SNR+3,:));
    BER_QAM(SNR+3) = BER_QAM(SNR+3)/num_bits;
    
    BER_8PSK(SNR+3) = num_bits - nnz(error_8PSK(SNR+3,:));
    BER_8PSK(SNR+3) = BER_8PSK(SNR+3)/num_bits;
end

%Calculating the theoritical BER
theo_BER_BPSK = 0.5 * erfc(sqrt(Eb./No));
theo_BER_QPSK = 0.5 * erfc(sqrt(Eb./No));
theo_BER_QAM  = (1.5/4) * erfc(sqrt(Eb_QAM./ (2.5.*No_QAM)));
theo_BER_8PSK = (1/3)*erfc(sqrt((3*Eb_8PSK)./No_8PSK)*(sin(pi/8)));

%Calculating the theoritical BER using another method for checking
berB = berawgn(SNR_range,'psk',2,'nondiff');
berQ = berawgn(SNR_range,'psk',4,'nondiff');
berQAM = berawgn(SNR_range,'qam',16,'nondiff');
ber8 = berawgn(SNR_range,'psk',8,'nondiff');

%plotting bit error rate with log scale versus SNR and drawing them on the 
%same graph along with the theoretical BER where 
figure('Name', 'BER Vs SNR');
%plotting the calculated BER as solid line
semilogy(SNR_range, BER_BPSK, 'r', SNR_range, BER_QPSK_1, 'b', ...
    SNR_range, BER_QAM, 'k', SNR_range, BER_8PSK, 'g');
hold on;
%plotting the theoretical BER as dashed line on the same figure
semilogy(SNR_range, theo_BER_BPSK, '-.r', SNR_range, theo_BER_QPSK,'--b', ...
    SNR_range, theo_BER_QAM,'--k', SNR_range, theo_BER_8PSK,'--g');
title ("BER Vs SNR");
xlabel("SNR: Eb/No (db)");
ylabel("BER");
xlim([-2,5]);
legend ('BER BPSK', 'BER QPSK', 'BER QAM', 'BER 8PSK', 'theo BER BPSK', ...
    'theo BER QPSK', 'theo BER QAM','theo BER 8PSK','Location','southwest')
grid on;

%plotting bit error rate with log scale versus SNR and for QPSK using grey 
%encoding and without using it and drawing them on the same graph
figure('Name', 'BER Vs SNR');
semilogy(SNR_range, BER_QPSK_1, 'b', SNR_range, BER_QPSK_2, 'r');
title ("BER Vs SNR");
xlabel("SNR: Eb/No (db)");
ylabel("BER");
xlim([-2,5]);
legend('BER QPSK using Gray coding', 'BER QPSK without Gray coding')
grid on;

%% Part2: BFSK
%--------------------------------5. BFSK-----------------------------------
%A. Mapping BFSK
BFSK_symbols = zeros(1,num_bits);
for k=1:num_bits
    if data_bits(k)
        BFSK_symbols(k)= 1i;
    else 
        BFSK_symbols(k)= 1;
    end
end

%//////////////////////////B. Adding noise//////////////////////////
%generating normally distributed noise with zero mean and variance=2
%due to adding two randn functions 
BFSK_noise = randn(1, num_bits) + (randn(1, num_bits))*1i;
% to have a symbol energy equal to one for BFSK
Eb_BFSK = 1;
%initialization of No used in the variance of noise (No/2) for BFSK
No_BFSK = zeros(1,8);

%initializing empty variables to store the scaled noise
scaled_noise_BFSK = zeros (8,num_bits);
%initalizing empty variables for demapped signal
demapped_BFSK = zeros(8,num_bits); 
%initializing empty variables for error
error_BFSK = zeros(8,num_bits); 
%initializing empty variable for bit error rate
BER_BFSK   = zeros(1,8);

%for loop that gets the required BER from the range of SNR [-2,5] db after
%adding the noise to the signal and demapping the received signal
for SNR = SNR_range
    %Calculating No for BFSK
    No_BFSK(SNR+3) = Eb_BFSK/(10^(SNR/10)); %BFSK No
    %making the noise with variance No/2 by multipying it by sqrt(No/2)
    scaled_noise_BFSK(SNR+3,:) = BFSK_noise*sqrt(No_BFSK(SNR+3)/2);
    %getting the transmitted signal with noise added to it
    v_BFSK = BFSK_symbols + scaled_noise_BFSK(SNR+3,:);
    
   %//////////////////////////C. Start demapping//////////////////////////
    for k=1:num_bits
        %demapping BFSK
        if angle(v_BFSK(k)) <= (pi/4) && angle(v_BFSK(k)) > (-3*pi/4) 
            demapped_BFSK(SNR+3,k) = 0;
        else
            demapped_BFSK(SNR+3,k) = 1;
        end
    end
    %//////////////////////////D. BER calculations////////////////////////
    %recognize the bits that differed between transmitted and received streams 
    error_BFSK(SNR+3,:) = demapped_BFSK(SNR+3,:) == data_bits;
    
    %counting the number of flipped bits and get the ratio
    %between num of bits that was flipped and the total number of bits in
    %each stream for each SNR to get bit error rate for each SNR value
    BER_BFSK(SNR+3) = num_bits - nnz(error_BFSK(SNR+3,:));
    BER_BFSK(SNR+3) = BER_BFSK(SNR+3)/num_bits;
end

%Calculating the theoritical BER
theo_BER_BFSK = 0.5 * erfc(sqrt(Eb_BFSK./(2*No_BFSK)));

%Calculating the theoritical BER using another method for checking
berBFSK = berawgn(SNR_range,'fsk',2,'coherent');

%plotting bit error rate with log scale versus SNR and drawing them on the 
%same graph along with the theoretical BER where 
figure('Name', 'BER Vs SNR');
%plotting the calculated BER as solid line and the theoretical BER as 
%dashed line on the same figure
semilogy(SNR_range, BER_BFSK, 'r',SNR_range, theo_BER_BFSK, '--b');
title ("BER Vs SNR");
xlabel("SNR: Eb/No (db)");
ylabel("BER");
xlim([-2,5]);
legend ('BER BFSK', 'theoritical BER BFSK','Location','southwest')
grid on;

%//////////////////////////E. Drawing PSD graph//////////////////////////
%initializing variables to be used
%number of realizations in the ensemble
ensemble_size = 20000; 
%number of bits in each realization
num_bits_PSD = 100; 
%pulse width to transmite one bit
bit_time = 70; 
%DAC sampling time
sampling_time = 10; 
%numbers of samples in on bit (7)
num_samples_per_bit = floor(bit_time / sampling_time); 
%total number of samples in one realization
num_samples = num_samples_per_bit*num_bits_PSD; 

%generating the baseband equivalent signal
S1_BB =sqrt(2*Eb_BFSK/(bit_time));
step = (bit_time/num_samples_per_bit);
t = 0: step :bit_time - (bit_time/num_samples_per_bit);
S2_BB = sqrt(2*Eb_BFSK/(bit_time))*cos(2*pi*(1/bit_time)*t)+ ...
    1i*sqrt(2*Eb_BFSK/(bit_time))*sin(2*pi*(1/bit_time)*t);

%fistly, generating the data as a matrix with nubmer of rows equal to the
%number of realization and number of coloumns represent the bits for each 
%one adding a bit for delay 
data_bits_PSD = randi([0,1], ensemble_size, num_bits_PSD+1); 
%secondly, storing the data after transforming it form bits to symbols
data_symbols=zeros(ensemble_size,(num_bits_PSD+1)*num_samples_per_bit);
for ii=1:ensemble_size
     for jj = 1:num_bits_PSD+1
        if data_bits_PSD(ii,jj)
            for k = 0 : num_samples_per_bit-1
                data_symbols(ii,num_samples_per_bit*(jj-1)+1+k)= S2_BB(k+1);
            end
        else
            for k = 0 : num_samples_per_bit-1
                data_symbols(ii,num_samples_per_bit*(jj-1)+1+k)= S1_BB;
            end
        end
     end
end

%Adding a random dalay at the start of each realization
%firstly, generating a delay less than the number of samples per bit for all 
%realizations
delay = randi([0,(num_samples_per_bit-1)],ensemble_size,1);
%defining a matrix to store the data after the delay
data = zeros(ensemble_size, num_samples);
%looping on all the realization and store the data from the end of the
%delay until getting all the samples
for i = 1:ensemble_size
    data(i,:)= data_symbols(i, delay(i)+1 : num_samples + delay(i));
end

%generate an empty matrix to store the result of statistical autocorrelation 
%in it
stat_autocorr = zeros(size(data(1,1:end)));
%loop on the coulmns of the data matrix to multiply the first coloumn by all 
%the other coulmns and take their sum and divided by the number of elemnts
%in that coloumn which is represented by the ensemble size to get the right
%sided autocorrelation from 0 to the number of samples
for i = 1 : num_samples
    stat_autocorr(1,i) = sum((conj(data(:,1)) .* data(:,i))) / ensemble_size;
end
%get the left side by fliping the previously calculated autocorrelation and
%concatenate them to get the final statistical autocorrelation
stat_autocorr = cat (2, conj(fliplr(stat_autocorr(2:num_samples))), stat_autocorr);

%Plotting PSD graph
k = -num_samples + 1: num_samples - 1;
fs = 100;  %Sampling frequency
figure('Name','PSD');
%on the other hand the y-axis is represented by the absolute value of the
%shifted fast fourier transform of the acf to represent the PSD
plot(k*fs/(2*num_samples),abs(fftshift(fft(stat_autocorr))));
title("PSD vs Frequency");
xlabel("Frequency(Hz)");
ylabel("PSD");
ylim([0,0.5]);