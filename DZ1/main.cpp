//-----------------------------------------------------------------------------
/* Домашнее задание №1. Задача №1 (нагревание стрежня).
 *
 * Реализация с использованием OpenACC.
 * Алгоритм на основе метода прогонки для неявной разностной схемы.
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
#define TWIDTH 5
#define XWIDTH 5
#define DWIDTH 20
//-----------------------------------------------------------------------------
float fi(int t) { return t*5; } // dT/dt
//-----------------------------------------------------------------------------
int main()
{
	ofstream fout("out");
	if (fout.fail()) {
		cout << "File error!" << endl;
		return 0;
	}
	fout << '#' << setw(TWIDTH-1) << "T" << ' ';
    fout << setw(XWIDTH) << "X" << ' ';
    fout << setw(DWIDTH) << "TEMP" << endl;

	const size_t NODES = 256; // Количество узлов
	const size_t TICKS = 256; // Количество итераций

	const float Hx = 0.5;     // Длина разбиений
	const float Ht = 0.05;     // Интервал времени

	float *data = new float[NODES]; // Найденные значения температуры

	float *ka_l = new float[NODES]; // Массив коэффициентов альфа на предыдущем шаге
	float *ka_n = new float[NODES]; // Массив коэффициентов альфа на текущем шаге
	float *kb_l = new float[NODES]; // Массив коэффициентов бета на предыдущем шаге
	float *kb_n = new float[NODES]; // Массив коэффициентов бета на текущем шаге

	const float A = - Ht / (Hx * Hx);         // Коэффициент при T[k+1, j+1]
	const float B = (2 * Ht) / (Hx * Hx) + 1; // Коэффициент при T[k+1, j]
	const float C = - Ht / (Hx * Hx);         // Коэффициент при T[k+1, j-1]
	const float E = 0;                        // Свободный член в правой части

	#pragma acc kernels loop independent
	// Первичная инициализация
	for (size_t i = 0; i < NODES; ++i)
		data[i] = 0;

	// Начинаем итерироваться

	for (size_t t = 0; t < TICKS; ++t) 
	{
		// Копируем коэффициенты с предыдущего шага
		memcpy(ka_l, ka_n, NODES*sizeof(float));
		memcpy(kb_l, kb_n, NODES*sizeof(float));

		// Из левого граничного условия 1-го рода
		ka_n[0] = 0;
		kb_n[0] = 0;

		#pragma acc kernels loop independent
		// Вычисляем значения альфа и бета для остальных точек
		for (int i = 1; i < NODES-1; ++i)
		{
			ka_n[i] = - A / (B + C * ka_l[i]);
			kb_n[i] = (E - C * kb_l[i]) / (B + C * ka_l[i]);
		}
		
		// Из правого граничного условия 2-го рода
		data[NODES-1] = (Hx * fi(t) + kb_n[NODES-2]) / (1 - ka_n[NODES-2]);

		// Из левого граничного условия
		data[0] = 0;

		// Прогон влево
		for (int i = NODES-2; i > 0; --i) // Вычисляем значения температур
		{
			data[i] = ka_n[i] * data[i+1] + kb_n[i];
		}
		
		// Пишем в таблицу в формате: %t %i %T
		for (int i = 0; i < NODES; ++i)
		{
			fout << setw(TWIDTH) << t << ' ';
            fout << setw(XWIDTH) << i << ' ';
            fout << setw(DWIDTH) << data[i] << endl;
		}

	}

	fout.close();

	delete data;
	delete ka_l;
	delete ka_n;
	delete kb_l;
	delete kb_n;

	return 0;
}
