function res = kalman(data)
  
  import Src.KalmanFilter.*
  
  % hlavna funkcia, ktora moderuje priebeh kalmanovho filtra.
  
  % Asserty a kontrola
  assert(~isempty(data));
  
  % Inicializacia a apriorne hodnoty nastavenia. Mozno mimo funkcie.
  %R = median(data);      % inicializacia prvej hodnoty stavoveho vektora
  R = data(1);      % inicializacia stavoveho vektora
  L = [R; data];         % vektor merani
  
  % apriori standard deviation
  sig = 1;          % Inicializacia kovariancnej matice
  
  %t = 0.5;
  Q = sig^2;           % presnost merania
  D = 1;               % stavova matica
  H = 1;               % transformacna matica. Matica planu. Derivacie modelu podla x = 1 ??? H
  F = .5;              % Na tejto konstante ladim 'hladkost' modelu
  
  xup(1,1) = R;       % Stavovy vektor
  Qvyr(1,1) = sig^2;   % Kovariancna matica
  Qe = F * sig^2 * F'; % Matica sumu: Obecny predpis. Kedze nas pripad je linearny,resp. 
                       % jednodimenzionalny, preto tento predpis ma len formalny charakter
  
  % Kalman filter
  xup = KalmanFilter(L, Q, D, H, xup, Qvyr,Qe);
  
  res = xup(2:end);
end