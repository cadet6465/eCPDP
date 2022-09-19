function S = LDAReg(train_data,train_label,n_class)
% the iscriminant constraint term based on the Fisher discrimination criterion

[~,N] = size(train_data);
m = mean(train_data,2);
m_i = [];
n_i = [];
for ii = 1 : n_class
    tr_dat_i = train_data(train_label == (ii - 1));
    temp = mean(tr_dat_i, 2);
    m_i = [m_i, temp];
    n_i = [n_i, size(tr_dat_i, 2)];
end

% between-class scatter matrix
Sb = zeros(size(m_i, 1));
for jj = 1 : n_class
    temp = m_i(:, jj) - m;
    temp2 = n_i(1,jj) * (temp * temp');
    Sb = Sb + temp2;
end

% total scatter matrix
mm = repmat(m,[1,N]);
St = (train_data-mm)*(train_data-mm)';

% between-class scatter matrix
Sw = St-Sb;

% the discriminant constraint term
S = Sw-Sb;
