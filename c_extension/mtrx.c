/* disable.c */

#define PY_SSIZE_T_CLEAN
#include <Python.h>


static double* matrix_mul(int m1r, int m1c, int m2r, int m2c, double* m1, double* m2) {
    double* product = (double*)PyMem_Calloc(m1r * m2c, sizeof(double));
    int vector_length = (m1c + m2r) / 2;

    for (int i = 0; i < m1r; i++) {
        for (int j = 0; j < m2c; j++) {
            for (int k = 0; k < vector_length; k++) {
                product[i * m2c + j] += m1[i*m1c+k] * m2[j+k*m2c];
            }
        }
    }
    return product;
}


static double* scalar_mul(int length, double* m, double a) {
    for (int i = 0; i < length; i++) {
        m[i] *= a;
    }
    return m;
}


static double* add(int length, double* m1, double* m2) {
    for (int i = 0; i < length; i++) {
        m1[i] += m2[i];
    }
    return m1;
}


static double* transpose(int r, int c, double* m) {
    for (int i = 0; i < r; i++) {
        for (int j = i+1; j < c; j++) {
            double temp = m[i*r+j];
            m[i*r+j] = m[j*r+i];
            m[j*r+i] = temp;
        }
    }
    return m;
}


static PyObject* matrix_mul_wrapper(PyObject* self, PyObject* args) {
    int m1r, m1c, m2r, m2c;
    PyObject *o1, *o2;

    if (!PyArg_ParseTuple(args, "iiiiOO", &m1r, &m1c, &m2r, &m2c, &o1, &o2)) {
        return NULL;
    }

    // Get fast sequences
    PyObject *seq1 = PySequence_Fast(o1, "");
    PyObject *seq2 = PySequence_Fast(o2, "");
    if (!seq1 || !seq2) {
        Py_XDECREF(seq1);
        Py_XDECREF(seq2);
        return NULL;
    }

    Py_ssize_t l1 = PySequence_Fast_GET_SIZE(seq1);
    Py_ssize_t l2 = PySequence_Fast_GET_SIZE(seq2);
    PyObject **items1 = PySequence_Fast_ITEMS(seq1);
    PyObject **items2 = PySequence_Fast_ITEMS(seq2);
    double* m1 = (double*)PyMem_Malloc(l1 * sizeof(double));
    double* m2 = (double*)PyMem_Malloc(l2 * sizeof(double));

    for (Py_ssize_t i = 0; i < l1; i++) {
        m1[i] = PyFloat_AsDouble(items1[i]);
        if (PyErr_Occurred()) goto fail;
    }
    for (Py_ssize_t i = 0; i < l2; i++) {
        m2[i] = PyFloat_AsDouble(items2[i]);
        if (PyErr_Occurred()) goto fail;
    }

    double* result;
    Py_BEGIN_ALLOW_THREADS
    result = matrix_mul(m1r, m1c, m2r, m2c, m1, m2);
    Py_END_ALLOW_THREADS

    PyObject* py_result = PyList_New(m1r * m2c);
    for (int i = 0; i < m1r * m2c; i++) {
        PyList_SetItem(py_result, i, PyFloat_FromDouble(result[i]));
    }

    PyMem_Free(m1);
    PyMem_Free(m2);
    PyMem_Free(result);
    Py_DECREF(seq1);
    Py_DECREF(seq2);
    return py_result;

fail:
    PyMem_Free(m1);
    PyMem_Free(m2);
    Py_DECREF(seq1);
    Py_DECREF(seq2);
    return NULL;
}


static PyObject* scalar_mul_wrapper(PyObject* self, PyObject* args) {
    double a;
    PyObject *o;

    if (!PyArg_ParseTuple(args, "Od", &o, &a)) {
        return NULL;
    }

    PyObject *seq = PySequence_Fast(o, "");
    if (!seq) {
        Py_XDECREF(seq);
        return NULL;
    }

    Py_ssize_t l = PySequence_Fast_GET_SIZE(seq);
    PyObject **items = PySequence_Fast_ITEMS(seq);
    double* m = (double*)PyMem_Malloc(l * sizeof(double));

    for (Py_ssize_t i = 0; i < l; i++) {
        m[i] = PyFloat_AsDouble(items[i]);
        if (PyErr_Occurred()) goto fail;
    }

    double* result;
    Py_BEGIN_ALLOW_THREADS
    result = scalar_mul(l, m, a);
    Py_END_ALLOW_THREADS

    PyObject* py_result = PyList_New(l);
    for (int i = 0; i < l; i++) {
        PyList_SetItem(py_result, i, PyFloat_FromDouble(result[i]));
    }

    PyMem_Free(m);
    Py_DECREF(seq);
    return py_result;

fail:
    PyMem_Free(m);
    Py_DECREF(seq);
    return NULL;
}


static PyObject* add_wrapper(PyObject* self, PyObject* args) {
    PyObject *o1, *o2;

    if (!PyArg_ParseTuple(args, "OO", &o1, &o2)) {
        return NULL;
    }

    PyObject *seq1 = PySequence_Fast(o1, "");
    PyObject *seq2 = PySequence_Fast(o2, "");
    if (!seq1 || !seq2) {
        Py_XDECREF(seq1);
        Py_XDECREF(seq2);
        return NULL;
    }

    Py_ssize_t l1 = PySequence_Fast_GET_SIZE(seq1);
    Py_ssize_t l2 = PySequence_Fast_GET_SIZE(seq2);
    PyObject **items1 = PySequence_Fast_ITEMS(seq1);
    PyObject **items2 = PySequence_Fast_ITEMS(seq2);
    double* m1 = (double*)PyMem_Malloc(l1 * sizeof(double));
    double* m2 = (double*)PyMem_Malloc(l2 * sizeof(double));

    for (Py_ssize_t i = 0; i < l1; i++) {
        m1[i] = PyFloat_AsDouble(items1[i]);
        if (PyErr_Occurred()) goto fail;
    }
    for (Py_ssize_t i = 0; i < l2; i++) {
        m2[i] = PyFloat_AsDouble(items2[i]);
        if (PyErr_Occurred()) goto fail;
    }

    int length = (l1+l2)/2;

    double* result;
    Py_BEGIN_ALLOW_THREADS
    result = add(length, m1, m2);
    Py_END_ALLOW_THREADS

    PyObject* py_result = PyList_New(length);
    for (int i = 0; i < length; i++) {
        PyList_SetItem(py_result, i, PyFloat_FromDouble(result[i]));
    }

    PyMem_Free(m1);
    PyMem_Free(m2);
    Py_DECREF(seq1);
    Py_DECREF(seq2);
    return py_result;

fail:
    PyMem_Free(m1);
    PyMem_Free(m2);
    Py_DECREF(seq1);
    Py_DECREF(seq2);
    return NULL;
}


static PyObject* transpose_wrapper(PyObject* self, PyObject* args) {
    int r, c;
    PyObject *o;

    if (!PyArg_ParseTuple(args, "iiO", &r,&c,&o)) {
        return NULL;
    }

    PyObject *seq = PySequence_Fast(o, "");
    if (!seq) {
        Py_XDECREF(seq);
        return NULL;
    }

    Py_ssize_t l = PySequence_Fast_GET_SIZE(seq);
    PyObject **items = PySequence_Fast_ITEMS(seq);
    double* m = (double*)PyMem_Malloc(l * sizeof(double));

    for (Py_ssize_t i = 0; i < l; i++) {
        m[i] = PyFloat_AsDouble(items[i]);
        if (PyErr_Occurred()) goto fail;
    }

    double* result;
    Py_BEGIN_ALLOW_THREADS
    result = transpose(r, c, m);
    Py_END_ALLOW_THREADS

    PyObject* py_result = PyList_New(l);
    for (int i = 0; i < l; i++) {
        PyList_SetItem(py_result, i, PyFloat_FromDouble(result[i]));
    }

    PyMem_Free(m);
    Py_DECREF(seq);
    return py_result;

fail:
    PyMem_Free(m);
    Py_DECREF(seq);
    return NULL;
}


static PyMethodDef ModuleMethods[] = {
    {"matrix_multiply", matrix_mul_wrapper, METH_VARARGS, ""},
    {"scalar_multiply", scalar_mul_wrapper, METH_VARARGS, ""},
    {"add", add_wrapper, METH_VARARGS, ""},
    {"transpose", transpose_wrapper, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef ModuleDefinition = {
    PyModuleDef_HEAD_INIT,
    "mtrx",                   /* name    */
    "matrix operations in C", /* doc     */
    -1,                       /* size    */
    ModuleMethods             /* methods */
};


PyMODINIT_FUNC PyInit_mtrx(void) {
    return PyModule_Create(&ModuleDefinition);
}
