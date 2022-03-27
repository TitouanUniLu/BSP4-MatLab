%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% inizializzazioni generali matlab
%%% function esterna: dente
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
close all
clear all
format long g


%stream = RandStream('mt19937ar',rng); % MATLAB's start-up settings
%stream = RandStream('mt19937ar');

%stream = RandStream('mt19937ar','Seed',sum(100*clock));
%RandStream.setDefaultStream(stream);





%%%%%%%%%%%%%%%%%%%%
%%% caricamento dati
%%%%%%%%%%%%%%%%%%%%
% Prezzi = xlsread('FTSE MIB.xlsx');
% Prezzi=xlsread('TFM2.xlsx');
% load TFM2.csv
% Prezzi = TFM2;
Prezzi = xlsread('TFM2.xls');
[n,numvar] = size(Prezzi);
Rendimenti = (Prezzi(2:end,:)-Prezzi(1:end-1,:))./Prezzi(1:end-1,:); % rendimenti percentuali
% Rendimenti = log(Prezzi(2:end,:)./Prezzi(1:end-1,:)); % rendimenti logaritmici
[T,numvar1] = size(Rendimenti);
maquanti = [150 100];

media = mean(Rendimenti);
adesso = num2str(round(10000*rem(now,1)));
tab1 = zeros(10,2);

pi_v = linspace(0.0001,0.00505578351944675,5); % vettore dei rendimenti minimo desiderato
pi_v = 0.0000967; % ***** ***** TEMPORANEO
npi = length(pi_v);

xfe = zeros(npi,1);

oggi = date; % for creating identifiers


for hij=1:npi
    pi = pi_v(hij);
    am = true; % am = true: portafoglio post-proc. ammissibile; am = false: portafoglio post-proc. non ammissibile
    rho_pre = zeros(1,2); % migliore rho pre-processamento
    rend_pre = zeros(1,2); % migliore rho pre-processamento
    diam_prepost = zeros(2,2); % migliori diametri pre- e post_processamento

    for giro = 1:2
        fprintf('%i\n', giro)
        c1 = 1.49618;
        c2 = 1.49618;
        w = 0.7298;
        chi = 1;
        a = chi*w;
        W = chi*((c1*0.5)+(c2*0.5));
        valore_max_W = 2*(a+1);

        epsilon = 1.0e-004; % penalizzazione violazione vincoli
        epsilon_or = epsilon; % pro benchmark
        percent_1 = 0.05; % quantity in (0,1) used to possibly update penalty parameters
        percent_2 = 0.90; % quantity in (0,1) used to possibly update penalty parameters
        Delta_k = 40; % number of iterations between two successive updates of penalty parameters
        Delta_k_rho = 20; % number of iterations between two successive updates of penalty parameter EPSILON
        max_weight = 10000; % maximum value of any penalty parameter
        min_weight = 0.0001; % minimum value of any penalty parameter
        max_epsilon = 1; % maximum value allowed for "epsilon"
        min_epsilon = 1.0e-15; % minimum value allowed for "epsilon"

        niter = maquanti(giro); % numero iterazioni

        nomone = 'CdTFP'; % identificatore file
        nomonef = nomone; % identificatore sottocartella
        nomone = [nomone '-' num2str(giro) '-' num2str(pi)]; % identificatore file
        nomoneD = nomone; % identificatore file
        nomoneFIT = nomone; % identificatore file
        nomoneRHO = nomone; % identificatore file
        nomoneEPSILON = nomone; % identificatore file
        nomonePESI = nomone; % identificatore file
        nomoneTAB1 = nomone; % identificatore file
        nomoneTAB2 = nomone; % identificatore file
        nomonePRTF = nomone; % identificatore file
        oggi
        oggi(1:2)
        adesso
        nomone = [nomone '-' oggi(1:2) oggi(4:6) oggi(8:11) '-' adesso '.txt'];
        nomone
    end
end


