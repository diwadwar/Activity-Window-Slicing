% Ustawienia wstępne
rootDir = '.'; % Ścieżka do głównego katalogu z danymi
outputFile = 'data.mat';

% Zakresy danych
personIds = 1:4;       % Osoby 1-4
gestureIds = 1:6;      % Gesty 1-6
repetitionIds = 1:5;   % Powtórzenia 1-5

% Konfiguracja wyboru cech
landmarkRows = 11:24;  % Wiersze landmarków (indeksowanie od 1)
coordCols = 1:3;       % Kolumny: x, y, z (pomijamy confidence)

% Liczba cech (14 landmarków * 3 współrzędne = 42 cechy)
numFeatures = length(landmarkRows) * length(coordCols);

% Inicjalizacja głównej struktury danych (Cell Array: Osoby x 1)
data = cell(length(personIds), 1);

fprintf('Rozpoczynam przetwarzanie danych (format: Klatki x Cechy)...\n');

% Pętla po osobach
for pIdx = 1:length(personIds)
    p = personIds(pIdx);
    
    % Obliczamy całkowitą liczbę próbek dla danej osoby
    numSamples = length(gestureIds) * length(repetitionIds);
    
    % Inicjalizacja macierzy dla osoby
    personData = cell(numSamples, 2);
    
    sampleCounter = 1;
    
    % Pętla po gestach
    for gIdx = 1:length(gestureIds)
        g = gestureIds(gIdx);
        labelStr = num2str(g); 
        
        % Pętla po powtórzeniach
        for rIdx = 1:length(repetitionIds)
            r = repetitionIds(rIdx);
            
            % Konstrukcja ścieżki
            currentPath = fullfile(rootDir, num2str(p), num2str(g), num2str(r));
            
            % Pobranie i sortowanie plików
            files = dir(fullfile(currentPath, '*.dat'));
            if isempty(files)
                warning('Brak plików w ścieżce: %s', currentPath);
                continue;
            end
            [~, sortedIdx] = sort({files.name});
            files = files(sortedIdx);
            
            numFrames = length(files);
            
            % ZMIANA 1: Alokacja macierzy [Klatki x Cechy]
            gestureSequence = zeros(numFrames, numFeatures);
            
            % Pętla po klatkach (plikach)
            for f = 1:numFrames
                filePath = fullfile(currentPath, files(f).name);
                
                try
                    rawData = load(filePath); 
                catch
                    warning('Błąd odczytu pliku: %s', filePath);
                    continue;
                end
                
                % Wybór danych (14x3)
                filteredData = rawData(landmarkRows, coordCols); 
                
                % ZMIANA 2: Spłaszczenie do wektora wierszowego [1 x 42]
                % Nadal używamy transpozycji filteredData', aby zachować kolejność (x1,y1,z1, x2,y2,z2...)
                % ale reshape robimy na 1 wiersz.
                featureRow = reshape(filteredData', 1, []);
                
                % ZMIANA 3: Przypisanie do wiersza f
                gestureSequence(f, :) = featureRow;
            end
            
            % Zapisanie przetworzonego gestu
            personData{sampleCounter, 1} = gestureSequence;
            personData{sampleCounter, 2} = labelStr;
            
            sampleCounter = sampleCounter + 1;
        end
    end
    
    % Przypisanie danych osoby do głównej struktury
    data{pIdx} = personData;
    fprintf('Przetworzono osobę %d/%d\n', p, length(personIds));
end

% Zapis do pliku
save(outputFile, 'data');
fprintf('Zakończono! Dane zapisane w pliku %s\n', outputFile);