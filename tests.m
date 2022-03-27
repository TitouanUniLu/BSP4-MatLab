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

