function xup = KalmanFilter(L, Q, D, A, xup, Qup, Qs)
  % INPUTS:
  %   L...
  %   Q...
  %   D...
  %   A...
  %   xup...
  %   Qup...
  %   Qs...
   
  % Dost zakladny Kalmanov filter. Motivacia z clanku
  % [1] Faragher, R. (2012): Understanding the Basis of the Kalman Filter Via a Simple and Intuitive
  %                          Derivation
  for k=1:length(L)-1
    
    % Predikce
    % rov. 3 v [1]
    xi=D * xup(k,1);            
    % rov. 4 v [1]  
    Qi=D * Qup(k,1) * D' + Qs;
    
    % Update
    % rov. 7 v [1]  
    K = Qi * A' / (A * Qi * A' + Q);                    
    % rov. 5 v [1]  
    xup(k+1,1) = xi + K * (L(k + 1, 1) - A * xi);
    % rov. 6 v [1]  
    Qup(k+1,1) = Qi - K * A * Qi;
  end
  
  
end