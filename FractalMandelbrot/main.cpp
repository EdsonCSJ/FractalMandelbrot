#include <mpi.h>
#include <SDL.h>
#include <iostream>

// Tamanho da janela
const int windowWidth = 800;
const int windowHeight = 600; 
const int maxIterations = 1000; // N�mero m�ximo de itera��es

// Intervalo para mapeamento das coordenadas da janela para o plano complexo
double minReal = -2.5; 
double maxReal = 1.0; 
double minImag = -1.5; 
double maxImag = 1.5; 

//Calcula o valor de itera��o de um pixel
int calculatePixel(int x, int y) {
    // Mapeia as coordenadas (x, y) para valores reais e imagin�rios no plano complexo
    double real = (x - windowWidth / 2.0) * (maxReal - minReal) / windowWidth + (maxReal + minReal) / 2.0;
    double imag = (y - windowHeight / 2.0) * (maxImag - minImag) / windowHeight + (maxImag + minImag) / 2.0;

    double zReal = 0.0;
    double zImag = 0.0;

    int iteration = 0;
    while (zReal * zReal + zImag * zImag <= 4.0 && iteration < maxIterations) {
        // Aplica a f�rmula do conjunto de Mandelbrot para calcular o pr�ximo valor de z
        double nextZReal = zReal * zReal - zImag * zImag + real;
        double nextZImag = 2.0 * zReal * zImag + imag;
        zReal = nextZReal;
        zImag = nextZImag;
        ++iteration;
    }

    return iteration;
}

//Renderiza os pixels de uma determinada regi�o
void renderPixels(int startRow, int endRow, Uint32* pixels) {
    for (int y = startRow; y < endRow; ++y) {
        for (int x = 0; x < windowWidth; ++x) {
            int iteration = calculatePixel(x, y);
            // Mapeia o valor do pixel para componentes RGB
            Uint8 r = iteration % 256; 
            Uint8 g = (iteration * 2) % 256; 
            Uint8 b = (iteration * 4) % 256; 

            pixels[(y - startRow) * windowWidth + x] = (255 << 24) | (r << 16) | (g << 8) | b; // Define o valor do pixel na imagem
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv); // Inicializa��o do MPI

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Obt�m o n�mero do processo atual
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Obt�m o n�mero total de processos

    double startTime = MPI_Wtime(); // Marca o tempo de in�cio da execu��o

    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;
    SDL_Texture* texture = nullptr;
    Uint32* pixels = nullptr;

    if (rank == 0) { // Apenas o processo com rank 0 realiza a inicializa��o da janela e da renderiza��o
        SDL_Init(SDL_INIT_VIDEO); // Inicializa��o do SDL para renderiza��o de gr�ficos

        // Cria��o da janela para exibir o fractal
        window = SDL_CreateWindow("Fractal de Mandelbrot", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
            windowWidth, windowHeight, SDL_WINDOW_SHOWN);

        renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED); // Cria��o do renderizador
        texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING,
            windowWidth, windowHeight); // Cria��o da textura para exibir os pixels

        pixels = new Uint32[windowWidth * windowHeight]; // Aloca��o de mem�ria para armazenar os pixels do fractal
    }

    int rowsPerProcess = windowHeight / size; // Quantidade de linhas do fractal processadas por cada processo
    int startRow = rank * rowsPerProcess; // Linha inicial do processo atual
    int endRow = (rank + 1) * rowsPerProcess; // Linha final do processo atual
    if (rank == size - 1) {
        endRow = windowHeight; // �ltimo processo pode ter um n�mero de linhas diferente dos outros
    }

    int localHeight = endRow - startRow; // Altura local do fractal para cada processo
    Uint32* localPixels = new Uint32[windowWidth * localHeight]; // Pixels locais para cada processo

    renderPixels(startRow, endRow, localPixels); // Renderiza os pixels da regi�o local

    MPI_Gather(localPixels, windowWidth * localHeight, MPI_UINT32_T,
        pixels, windowWidth * localHeight, MPI_UINT32_T,
        0, MPI_COMM_WORLD); // Coleta os pixels de todos os processos no processo 0

    delete[] localPixels; // Libera a mem�ria dos pixels locais

    if (rank == 0) { // Apenas o processo com rank 0 atualiza a textura, renderiza e exibe a janela
        SDL_UpdateTexture(texture, NULL, pixels, windowWidth * sizeof(Uint32)); // Atualiza a textura com os pixels

        SDL_RenderClear(renderer); // Limpa o renderizador
        SDL_RenderCopy(renderer, texture, NULL, NULL); // Copia a textura para o renderizador
        SDL_RenderPresent(renderer); // Renderiza na janela

        bool quit = false;
        SDL_Event event;

        double endTime = MPI_Wtime(); // Marca o tempo de fim da execu��o
        double elapsedTime = endTime - startTime; // Calcula o tempo total de execu��o
        std::cout << "Tempo de renderizacao: " << elapsedTime << " segundos" << std::endl; // Exibe o tempo total de execu��o

        while (!quit) {
            while (SDL_PollEvent(&event) != 0) {
                if (event.type == SDL_QUIT) {
                    quit = true; // Finaliza o loop se o evento de fechar a janela for detectado
                }
            }
        }

        delete[] pixels; 

        SDL_DestroyTexture(texture); 
        SDL_DestroyRenderer(renderer); 
        SDL_DestroyWindow(window); 

        SDL_Quit(); 
    }

    MPI_Finalize();

    return 0;
}
