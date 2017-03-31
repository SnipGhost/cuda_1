//-----------------------------------------------------------------------------
/* Домашнее задание №1. Задача №1 (нагревание стрежня).
 *
 * Реализация с использованием OpenACC.
 * Алгоритм на основе последовательной итерации для явной разностной схемы.
 *
 *                                  By SnipGhost (Михаил Кучеренко), 28.03.2017
*/
//-----------------------------------------------------------------------------
#include <iostream>
#include <fstream>
#include <cstring>
#include <iomanip>
#include <openacc.h>
using namespace std;
//-----------------------------------------------------------------------------
#define INCREMENT 5 // dT/dt
#define H_TIME 0.05 // Ht
#define H_XLEN 0.5  // Hx
//-----------------------------------------------------------------------------
#define TWIDTH 5    // Ширина поля вывода для времени
#define XWIDTH 5    // Ширина поля вывода для номера узла
#define DWIDTH 20   // Ширина поля вывода для температуры узла
//-----------------------------------------------------------------------------
int main()
{
	const int NODES = 64; // Количество узлов
	const int TICKS = 256; // Количество итераций

	float *data = new float[NODES];
	float *buff = new float[NODES];

	ofstream fout("out");
	if (fout.fail()) {
		cout << "File error!" << endl;
		return 0;
	}
	fout << '#' << setw(XWIDTH-1) << "X" << ' ';
    fout << setw(TWIDTH) << "T" << ' ';
    fout << setw(DWIDTH) << "TEMP" << endl;

#pragma acc kernels loop independent
	for (int i = 0; i < NODES; ++i)
		data[i] = 0;
	
	memcpy(buff, data, NODES*sizeof(float));

	for (int t = 0; t < TICKS; ++t) 
	{

#pragma acc kernels loop independent
		for (int i = 0; i < NODES; ++i)
		{
			if (i == 0)
				buff[0] = 0;
			else if (i == NODES-1)
				buff[i] += INCREMENT;
			else
				buff[i] += (data[i+1] - 2*data[i] + data[i-1]) * H_TIME / (H_XLEN * H_XLEN);
		}

		memcpy(data, buff, NODES*sizeof(float));

		for (int i = 0; i < NODES; ++i)
		{
			fout << setw(XWIDTH) << i << ' ';			
			fout << setw(TWIDTH) << t << ' ';            
            fout << setw(DWIDTH) << data[i] << endl;
		}

	}

	fout.close();	

	delete data;
	delete buff;

	return 0;
}
