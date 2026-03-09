#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>   // memset

static PyObject* convolve_forward(PyObject *self, PyObject *args) {
    // Buffers
    Py_buffer xb = {0}, Wb = {0}, bb = {0};

    // x args3: (buf, x_off, x_s0, x_s1, x_s2, x_st0, x_st1, x_st2)
    Py_ssize_t x_off, x_s0, x_s1, x_s2, x_st0, x_st1, x_st2;

    // W args4: (buf, W_off, W_s0, W_s1, W_s2, W_s3, W_st0, W_st1, W_st2, W_st3)
    Py_ssize_t W_off, W_s0, W_s1, W_s2, W_s3, W_st0, W_st1, W_st2, W_st3;

    // b args1: (buf, b_off, b_s0, b_st0)
    Py_ssize_t b_off, b_s0, b_st0;

    // Extras (optional but matches your proposed call)
    Py_ssize_t cout_arg, kh_arg, kw_arg;

    // Parse exactly: *x.args3(), *W.args4(), *b.args1(), cout, kh, kw
    if (!PyArg_ParseTuple(
            args,
            "y*nnnnnnn"   // x: 1 buffer + 7 Py_ssize_t
            "y*nnnnnnnnn" // W: 1 buffer + 9 Py_ssize_t
            "y*nnn"       // b: 1 buffer + 3 Py_ssize_t
            "nnn",        // extras: cout, kh, kw
            &xb, &x_off, &x_s0, &x_s1, &x_s2, &x_st0, &x_st1, &x_st2,
            &Wb, &W_off, &W_s0, &W_s1, &W_s2, &W_s3, &W_st0, &W_st1, &W_st2, &W_st3,
            &bb, &b_off, &b_s0, &b_st0,
            &cout_arg, &kh_arg, &kw_arg
        )) {
        return NULL;
    }

    // Validate buffers are doubles and contiguous enough for pointer arithmetic
    // We rely on memoryview(array('d')) -> format "d" and itemsize 8.
    if (xb.itemsize != (Py_ssize_t)sizeof(double) ||
        Wb.itemsize != (Py_ssize_t)sizeof(double) ||
        bb.itemsize != (Py_ssize_t)sizeof(double)) {
        PyErr_SetString(PyExc_TypeError, "Expected buffers of type 'double' (memoryview of array('d'))");
        goto fail;
    }

    double *xdata = (double*)xb.buf;
    double *Wdata = (double*)Wb.buf;
    double *bdata = (double*)bb.buf;

    // Interpret shapes
    // x: (Cin, H, W)
    Py_ssize_t Cin = x_s0;
    Py_ssize_t H   = x_s1;
    Py_ssize_t Ww  = x_s2;

    // W: (Cout, Cin, Kh, Kw)
    Py_ssize_t Cout = W_s0;
    Py_ssize_t W_Cin = W_s1;
    Py_ssize_t Kh = W_s2;
    Py_ssize_t Kw = W_s3;

    // b: (Cout,)
    if (b_s0 != Cout) {
        PyErr_SetString(PyExc_ValueError, "b.shape[0] must equal W.shape[0] (Cout)");
        goto fail;
    }
    if (W_Cin != Cin) {
        PyErr_SetString(PyExc_ValueError, "W.shape[1] (Cin) must equal x.shape[0] (Cin)");
        goto fail;
    }

    // Optional consistency checks vs passed cout/kh/kw
    if (cout_arg != Cout || kh_arg != Kh || kw_arg != Kw) {
        PyErr_SetString(PyExc_ValueError, "cout/kh/kw args do not match kernel shape (W)");
        goto fail;
    }

    // Output dims: valid convolution
    Py_ssize_t Hout = H - Kh + 1;
    Py_ssize_t Wout = Ww - Kw + 1;
    if (Hout <= 0 || Wout <= 0) {
        PyErr_SetString(PyExc_ValueError, "Invalid output shape (Hout/Wout <= 0)");
        goto fail;
    }

    // Allocate output: array('d') of size Cout*Hout*Wout, filled with zeros
    Py_ssize_t out_n = Cout * Hout * Wout;
    Py_ssize_t out_bytes = out_n * (Py_ssize_t)sizeof(double);

    PyObject *array_mod = PyImport_ImportModule("array");
    if (!array_mod) goto fail;
    PyObject *array_type = PyObject_GetAttrString(array_mod, "array");
    Py_DECREF(array_mod);
    if (!array_type) goto fail;

    PyObject *out_arr = PyObject_CallFunction(array_type, "s", "d"); // array('d')
    Py_DECREF(array_type);
    if (!out_arr) goto fail;

    // Create a zeroed bytes object and load it into array via frombytes (fast; no Python floats)
    PyObject *zero_bytes = PyBytes_FromStringAndSize(NULL, out_bytes);
    if (!zero_bytes) { Py_DECREF(out_arr); goto fail; }
    memset(PyBytes_AS_STRING(zero_bytes), 0, (size_t)out_bytes);

    PyObject *res = PyObject_CallMethod(out_arr, "frombytes", "O", zero_bytes);
    Py_DECREF(zero_bytes);
    if (!res) { Py_DECREF(out_arr); goto fail; }
    Py_DECREF(res);

    // Get writable buffer to output
    Py_buffer yb = {0};
    if (PyObject_GetBuffer(out_arr, &yb, PyBUF_WRITABLE) != 0) {
        Py_DECREF(out_arr);
        goto fail;
    }
    double *ydata = (double*)yb.buf;

    // Main convolution
    // Using passed strides and offsets so it works with Tensor views/slices.
    // x index: x_off + ic*x_st0 + iy*x_st1 + ix*x_st2
    // W index: W_off + oc*W_st0 + ic*W_st1 + ky*W_st2 + kx*W_st3
    // b index: b_off + oc*b_st0
    for (Py_ssize_t oc = 0; oc < Cout; oc++) {
        double b_oc = bdata[b_off + oc * b_st0];
        Py_ssize_t W_oc_base = W_off + oc * W_st0;
        Py_ssize_t y_oc_base = oc * (Hout * Wout);

        for (Py_ssize_t oy = 0; oy < Hout; oy++) {
            Py_ssize_t y_row_base = y_oc_base + oy * Wout;

            for (Py_ssize_t ox = 0; ox < Wout; ox++) {
                double acc = b_oc;

                for (Py_ssize_t ic = 0; ic < Cin; ic++) {
                    Py_ssize_t x_ic_base = x_off + ic * x_st0;
                    Py_ssize_t W_ocic_base = W_oc_base + ic * W_st1;

                    for (Py_ssize_t ky = 0; ky < Kh; ky++) {
                        Py_ssize_t x_row = x_ic_base + (oy + ky) * x_st1;
                        Py_ssize_t W_row = W_ocic_base + ky * W_st2;

                        for (Py_ssize_t kx = 0; kx < Kw; kx++) {
                            Py_ssize_t xi = x_row + (ox + kx) * x_st2;
                            Py_ssize_t wi = W_row + kx * W_st3;
                            acc += xdata[xi] * Wdata[wi];
                        }
                    }
                }

                ydata[y_row_base + ox] = acc;
            }
        }
    }

    PyBuffer_Release(&yb);
    PyBuffer_Release(&xb);
    PyBuffer_Release(&Wb);
    PyBuffer_Release(&bb);

    return out_arr;

fail:
    PyBuffer_Release(&xb);
    PyBuffer_Release(&Wb);
    PyBuffer_Release(&bb);
    return NULL;
}

static int ensure_double_buf(Py_buffer *b, const char *name) {
    if (b->itemsize != (Py_ssize_t)sizeof(double)) {
        PyErr_Format(PyExc_TypeError, "%s: expected buffer of doubles (itemsize=8)", name);
        return 0;
    }
    return 1;
}

static PyObject* convolve_backward(PyObject *self, PyObject *args) {
    // Buffers
    Py_buffer xb = {0}, dyb = {0}, Wb = {0}, dWb = {0}, dbb = {0};

    // x args3: (buf, off, s0,s1,s2, st0,st1,st2)
    Py_ssize_t x_off, x_s0, x_s1, x_s2, x_st0, x_st1, x_st2;
    // dy args3
    Py_ssize_t dy_off, dy_s0, dy_s1, dy_s2, dy_st0, dy_st1, dy_st2;
    // W args4: (buf, off, s0,s1,s2,s3, st0,st1,st2,st3)
    Py_ssize_t W_off, W_s0, W_s1, W_s2, W_s3, W_st0, W_st1, W_st2, W_st3;
    // dW args4 (writable)
    Py_ssize_t dW_off, dW_s0, dW_s1, dW_s2, dW_s3, dW_st0, dW_st1, dW_st2, dW_st3;
    // db args1 (writable): (buf, off, s0, st0)
    Py_ssize_t db_off, db_s0, db_st0;

    // extras
    Py_ssize_t cout_arg, kh_arg, kw_arg;

    // Parse: *x.args3(), *dy.args3(), *W.args4(), *dW.args4(), *db.args1(), cout, kh, kw
    if (!PyArg_ParseTuple(
            args,
            "y*nnnnnnn"    // x: 1 buffer + 7 ints
            "y*nnnnnnn"    // dy: 1 buffer + 7 ints
            "y*nnnnnnnnn"  // W: 1 buffer + 9 ints
            "y*nnnnnnnnn"  // dW: 1 buffer + 9 ints
            "y*nnn"        // db: 1 buffer + 3 ints
            "nnn",         // extras
            &xb, &x_off, &x_s0, &x_s1, &x_s2, &x_st0, &x_st1, &x_st2,
            &dyb, &dy_off, &dy_s0, &dy_s1, &dy_s2, &dy_st0, &dy_st1, &dy_st2,
            &Wb, &W_off, &W_s0, &W_s1, &W_s2, &W_s3, &W_st0, &W_st1, &W_st2, &W_st3,
            &dWb, &dW_off, &dW_s0, &dW_s1, &dW_s2, &dW_s3, &dW_st0, &dW_st1, &dW_st2, &dW_st3,
            &dbb, &db_off, &db_s0, &db_st0,
            &cout_arg, &kh_arg, &kw_arg
        )) {
        return NULL;
    }

    // Validate dtype (double)
    if (!ensure_double_buf(&xb, "x") ||
        !ensure_double_buf(&dyb, "dy") ||
        !ensure_double_buf(&Wb, "W") ||
        !ensure_double_buf(&dWb, "dW") ||
        !ensure_double_buf(&dbb, "db")) {
        goto fail;
    }

    if (dWb.readonly) { PyErr_SetString(PyExc_TypeError, "dW buffer must be writable"); goto fail; }
    if (dbb.readonly) { PyErr_SetString(PyExc_TypeError, "db buffer must be writable"); goto fail; }

    double *xdata  = (double*)xb.buf;
    double *dydata = (double*)dyb.buf;
    double *Wdata  = (double*)Wb.buf;
    double *dWdata = (double*)dWb.buf;
    double *dbdata = (double*)dbb.buf;

    // Shapes
    Py_ssize_t Cin  = x_s0;
    Py_ssize_t H    = x_s1;
    Py_ssize_t Ww   = x_s2;

    Py_ssize_t Cout = dy_s0;
    Py_ssize_t Hout = dy_s1;
    Py_ssize_t Wout = dy_s2;

    Py_ssize_t W_Cout = W_s0;
    Py_ssize_t W_Cin  = W_s1;
    Py_ssize_t Kh     = W_s2;
    Py_ssize_t Kw     = W_s3;

    // Consistency checks
    if (Cout != W_Cout || Cin != W_Cin) {
        PyErr_SetString(PyExc_ValueError, "Shape mismatch: dy/W or x/W channel dims");
        goto fail;
    }
    if (Kh != kh_arg || Kw != kw_arg || Cout != cout_arg) {
        PyErr_SetString(PyExc_ValueError, "cout/kh/kw args do not match dy/W shapes");
        goto fail;
    }
    if (db_s0 != Cout) {
        PyErr_SetString(PyExc_ValueError, "db.shape[0] must equal Cout");
        goto fail;
    }
    // Validate valid-conv relation (optional but helpful)
    if (Hout != (H - Kh + 1) || Wout != (Ww - Kw + 1)) {
        PyErr_SetString(PyExc_ValueError, "dy shape does not match valid conv output shape from x and kernel");
        goto fail;
    }

    // Allocate dx = zeros(Cin,H,W) as array('d') using frombytes (fast)
    Py_ssize_t dx_n = Cin * H * Ww;
    Py_ssize_t dx_bytes = dx_n * (Py_ssize_t)sizeof(double);

    PyObject *array_mod = PyImport_ImportModule("array");
    if (!array_mod) goto fail;
    PyObject *array_type = PyObject_GetAttrString(array_mod, "array");
    Py_DECREF(array_mod);
    if (!array_type) goto fail;

    PyObject *dx_arr = PyObject_CallFunction(array_type, "s", "d"); // array('d')
    Py_DECREF(array_type);
    if (!dx_arr) goto fail;

    PyObject *zero_bytes = PyBytes_FromStringAndSize(NULL, dx_bytes);
    if (!zero_bytes) { Py_DECREF(dx_arr); goto fail; }
    memset(PyBytes_AS_STRING(zero_bytes), 0, (size_t)dx_bytes);

    PyObject *res = PyObject_CallMethod(dx_arr, "frombytes", "O", zero_bytes);
    Py_DECREF(zero_bytes);
    if (!res) { Py_DECREF(dx_arr); goto fail; }
    Py_DECREF(res);

    Py_buffer dxb = {0};
    if (PyObject_GetBuffer(dx_arr, &dxb, PyBUF_WRITABLE) != 0) {
        Py_DECREF(dx_arr);
        goto fail;
    }
    double *dxdata = (double*)dxb.buf;

    // ---- db: db[oc] += sum_{oy,ox} dy[oc,oy,ox]
    for (Py_ssize_t oc = 0; oc < Cout; oc++) {
        double acc = 0.0;
        Py_ssize_t dy_oc_base = dy_off + oc * dy_st0;

        for (Py_ssize_t oy = 0; oy < Hout; oy++) {
            Py_ssize_t dy_row = dy_oc_base + oy * dy_st1;
            for (Py_ssize_t ox = 0; ox < Wout; ox++) {
                acc += dydata[dy_row + ox * dy_st2];
            }
        }
        dbdata[db_off + oc * db_st0] += acc;
    }

    // ---- dW and dx
    // dW[oc,ic,ky,kx] += x[ic,oy+ky,ox+kx] * dy[oc,oy,ox]
    // dx[ic,oy+ky,ox+kx] += W[oc,ic,ky,kx] * dy[oc,oy,ox]
    for (Py_ssize_t oc = 0; oc < Cout; oc++) {
        Py_ssize_t dy_oc_base = dy_off + oc * dy_st0;
        Py_ssize_t W_oc_base  = W_off  + oc * W_st0;
        Py_ssize_t dW_oc_base = dW_off + oc * dW_st0;

        for (Py_ssize_t oy = 0; oy < Hout; oy++) {
            Py_ssize_t dy_row = dy_oc_base + oy * dy_st1;

            for (Py_ssize_t ox = 0; ox < Wout; ox++) {
                double g = dydata[dy_row + ox * dy_st2];

                for (Py_ssize_t ic = 0; ic < Cin; ic++) {
                    Py_ssize_t x_ic_base  = x_off  + ic * x_st0;
                    Py_ssize_t dx_ic_base = ic * (H * Ww); // contiguous layout for dx_arr we allocated

                    Py_ssize_t W_ocic_base  = W_oc_base  + ic * W_st1;
                    Py_ssize_t dW_ocic_base = dW_oc_base + ic * dW_st1;

                    for (Py_ssize_t ky = 0; ky < Kh; ky++) {
                        Py_ssize_t iy = oy + ky;

                        Py_ssize_t x_row = x_ic_base + iy * x_st1;
                        Py_ssize_t dx_row = dx_ic_base + iy * Ww; // because dx is contiguous Cin-major then row-major

                        Py_ssize_t W_row  = W_ocic_base  + ky * W_st2;
                        Py_ssize_t dW_row = dW_ocic_base + ky * dW_st2;

                        for (Py_ssize_t kx = 0; kx < Kw; kx++) {
                            Py_ssize_t ix = ox + kx;

                            Py_ssize_t xidx  = x_row + ix * x_st2;
                            Py_ssize_t dxidx = dx_row + ix; // contiguous last-dim stride=1

                            Py_ssize_t widx  = W_row  + kx * W_st3;
                            Py_ssize_t dwidx = dW_row + kx * dW_st3;

                            dWdata[dwidx] += xdata[xidx] * g;
                            dxdata[dxidx] += Wdata[widx] * g;
                        }
                    }
                }
            }
        }
    }

    PyBuffer_Release(&dxb);
    PyBuffer_Release(&xb);
    PyBuffer_Release(&dyb);
    PyBuffer_Release(&Wb);
    PyBuffer_Release(&dWb);
    PyBuffer_Release(&dbb);

    return dx_arr;

fail:
    PyBuffer_Release(&xb);
    PyBuffer_Release(&dyb);
    PyBuffer_Release(&Wb);
    PyBuffer_Release(&dWb);
    PyBuffer_Release(&dbb);
    return NULL;
}

static PyObject* make_zeroed_array_d(Py_ssize_t n_doubles) {
    Py_ssize_t nbytes = n_doubles * (Py_ssize_t)sizeof(double);

    PyObject *array_mod = PyImport_ImportModule("array");
    if (!array_mod) return NULL;
    PyObject *array_type = PyObject_GetAttrString(array_mod, "array");
    Py_DECREF(array_mod);
    if (!array_type) return NULL;

    PyObject *arr = PyObject_CallFunction(array_type, "s", "d"); // array('d')
    Py_DECREF(array_type);
    if (!arr) return NULL;

    PyObject *zero_bytes = PyBytes_FromStringAndSize(NULL, nbytes);
    if (!zero_bytes) { Py_DECREF(arr); return NULL; }
    memset(PyBytes_AS_STRING(zero_bytes), 0, (size_t)nbytes);

    PyObject *res = PyObject_CallMethod(arr, "frombytes", "O", zero_bytes);
    Py_DECREF(zero_bytes);
    if (!res) { Py_DECREF(arr); return NULL; }
    Py_DECREF(res);

    return arr;
}

static PyObject* dense_forward(PyObject *self, PyObject *args) {
    Py_buffer xb = {0}, Wb = {0}, bb = {0};
    Py_ssize_t x_off, x_s0, x_st0;
    Py_ssize_t W_off, W_s0, W_s1, W_st0, W_st1;
    Py_ssize_t b_off, b_s0, b_st0;
    Py_ssize_t din, dout;

    if (!PyArg_ParseTuple(
            args,
            "y*nnn"        // x
            "y*nnnnn"      // W
            "y*nnn"        // b
            "nn",          // din, dout
            &xb, &x_off, &x_s0, &x_st0,
            &Wb, &W_off, &W_s0, &W_s1, &W_st0, &W_st1,
            &bb, &b_off, &b_s0, &b_st0,
            &din, &dout
        )) {
        return NULL;
    }

    if (!ensure_double_buf(&xb, "x") || !ensure_double_buf(&Wb, "W") || !ensure_double_buf(&bb, "b")) goto fail;

    // shape checks
    if (x_s0 != din || W_s0 != din || W_s1 != dout || b_s0 != dout) {
        PyErr_SetString(PyExc_ValueError, "Dense.forward shape mismatch");
        goto fail;
    }

    double *xdata = (double*)xb.buf;
    double *Wdata = (double*)Wb.buf;
    double *bdata = (double*)bb.buf;

    PyObject *z_arr = make_zeroed_array_d(dout);
    if (!z_arr) goto fail;

    Py_buffer zb = {0};
    if (PyObject_GetBuffer(z_arr, &zb, PyBUF_WRITABLE) != 0) { Py_DECREF(z_arr); goto fail; }
    double *zdata = (double*)zb.buf;

    // z[j] = b[j] + sum_k x[k] * W[k,j]
    for (Py_ssize_t j = 0; j < dout; j++) {
        double acc = bdata[b_off + j * b_st0];

        Py_ssize_t W_col_base = W_off + j * W_st1; // base of W[:, j]
        for (Py_ssize_t k = 0; k < din; k++) {
            acc += xdata[x_off + k * x_st0] * Wdata[W_col_base + k * W_st0];
        }
        zdata[j] = acc; // contiguous output
    }

    PyBuffer_Release(&zb);
    PyBuffer_Release(&xb);
    PyBuffer_Release(&Wb);
    PyBuffer_Release(&bb);
    return z_arr;

fail:
    PyBuffer_Release(&xb);
    PyBuffer_Release(&Wb);
    PyBuffer_Release(&bb);
    return NULL;
}

static PyObject* dense_backward(PyObject *self, PyObject *args) {
    Py_buffer xb = {0}, dzb = {0}, Wb = {0}, dWb = {0}, dbb = {0};
    Py_ssize_t x_off, x_s0, x_st0;
    Py_ssize_t dz_off, dz_s0, dz_st0;
    Py_ssize_t W_off, W_s0, W_s1, W_st0, W_st1;
    Py_ssize_t dW_off, dW_s0, dW_s1, dW_st0, dW_st1;
    Py_ssize_t db_off, db_s0, db_st0;
    Py_ssize_t din, dout;

    if (!PyArg_ParseTuple(
            args,
            "y*nnn"        // x
            "y*nnn"        // dz
            "y*nnnnn"      // W
            "y*nnnnn"      // dW
            "y*nnn"        // db
            "nn",          // din, dout
            &xb, &x_off, &x_s0, &x_st0,
            &dzb, &dz_off, &dz_s0, &dz_st0,
            &Wb, &W_off, &W_s0, &W_s1, &W_st0, &W_st1,
            &dWb, &dW_off, &dW_s0, &dW_s1, &dW_st0, &dW_st1,
            &dbb, &db_off, &db_s0, &db_st0,
            &din, &dout
        )) {
        return NULL;
    }

    if (!ensure_double_buf(&xb, "x") || !ensure_double_buf(&dzb, "dz") ||
        !ensure_double_buf(&Wb, "W") || !ensure_double_buf(&dWb, "dW") || !ensure_double_buf(&dbb, "db")) goto fail;

    if (dWb.readonly) { PyErr_SetString(PyExc_TypeError, "dW buffer must be writable"); goto fail; }
    if (dbb.readonly) { PyErr_SetString(PyExc_TypeError, "db buffer must be writable"); goto fail; }

    // shape checks
    if (x_s0 != din || dz_s0 != dout ||
        W_s0 != din || W_s1 != dout ||
        dW_s0 != din || dW_s1 != dout ||
        db_s0 != dout) {
        PyErr_SetString(PyExc_ValueError, "Dense.backward shape mismatch");
        goto fail;
    }

    double *xdata  = (double*)xb.buf;
    double *dzdata = (double*)dzb.buf;
    double *Wdata  = (double*)Wb.buf;
    double *dWdata = (double*)dWb.buf;
    double *dbdata = (double*)dbb.buf;

    // dx output (contiguous)
    PyObject *dx_arr = make_zeroed_array_d(din);
    if (!dx_arr) goto fail;

    Py_buffer dxb = {0};
    if (PyObject_GetBuffer(dx_arr, &dxb, PyBUF_WRITABLE) != 0) { Py_DECREF(dx_arr); goto fail; }
    double *dxdata = (double*)dxb.buf;

    // db[j] += dz[j]
    for (Py_ssize_t j = 0; j < dout; j++) {
        double g = dzdata[dz_off + j * dz_st0];
        dbdata[db_off + j * db_st0] += g;
    }

    // dW and dx
    // dW[k,j] += x[k] * dz[j]
    // dx[k]   += W[k,j] * dz[j]
    for (Py_ssize_t j = 0; j < dout; j++) {
        double g = dzdata[dz_off + j * dz_st0];

        Py_ssize_t W_col_base  = W_off  + j * W_st1;
        Py_ssize_t dW_col_base = dW_off + j * dW_st1;

        for (Py_ssize_t k = 0; k < din; k++) {
            double xk = xdata[x_off + k * x_st0];
            dWdata[dW_col_base + k * dW_st0] += xk * g;
            dxdata[k] += Wdata[W_col_base + k * W_st0] * g;
        }
    }

    PyBuffer_Release(&dxb);
    PyBuffer_Release(&xb);
    PyBuffer_Release(&dzb);
    PyBuffer_Release(&Wb);
    PyBuffer_Release(&dWb);
    PyBuffer_Release(&dbb);
    return dx_arr;

fail:
    PyBuffer_Release(&xb);
    PyBuffer_Release(&dzb);
    PyBuffer_Release(&Wb);
    PyBuffer_Release(&dWb);
    PyBuffer_Release(&dbb);
    return NULL;
}

static PyObject* pool_forward_max(PyObject *self, PyObject *args) {
    Py_buffer xb = {0};
    Py_ssize_t x_off, C, H, W, xs0, xs1, xs2;
    Py_ssize_t ph, pw, s;

    if (!PyArg_ParseTuple(
            args,
            "y*nnnnnnn"  // *x.args3()
            "nnn",       // ph, pw, s
            &xb, &x_off, &C, &H, &W, &xs0, &xs1, &xs2,
            &ph, &pw, &s
        )) return NULL;

    if (!ensure_double_buf(&xb, "x")) goto fail;

    if (ph <= 0 || pw <= 0 || s <= 0) {
        PyErr_SetString(PyExc_ValueError, "ph/pw/stride must be > 0");
        goto fail;
    }

    Py_ssize_t Hout = (H - ph) / s + 1;
    Py_ssize_t Wout = (W - pw) / s + 1;
    if (Hout <= 0 || Wout <= 0) {
        PyErr_SetString(PyExc_ValueError, "Invalid output shape for pooling");
        goto fail;
    }

    double *xdata = (double*)xb.buf;

    Py_ssize_t out_n = C * Hout * Wout;
    PyObject *y_arr = make_zeroed_array_d(out_n);
    if (!y_arr) goto fail;

    PyObject *am_arr = make_zeroed_array_d(out_n);
    if (!am_arr) { Py_DECREF(y_arr); goto fail; }

    Py_buffer yb = {0}, amb = {0};
    if (PyObject_GetBuffer(y_arr, &yb, PyBUF_WRITABLE) != 0) { Py_DECREF(y_arr); Py_DECREF(am_arr); goto fail; }
    if (PyObject_GetBuffer(am_arr, &amb, PyBUF_WRITABLE) != 0) { PyBuffer_Release(&yb); Py_DECREF(y_arr); Py_DECREF(am_arr); goto fail; }

    double *ydata  = (double*)yb.buf;
    double *amdata = (double*)amb.buf;

    // Output layout we produce: contiguous (C, Hout, Wout) with strides:
    // oc_stride = Hout*Wout, oy_stride = Wout, ox_stride = 1
    Py_ssize_t oc_stride = Hout * Wout;

    for (Py_ssize_t c = 0; c < C; c++) {
        Py_ssize_t x_c = x_off + c * xs0;
        Py_ssize_t y_c = c * oc_stride;
        Py_ssize_t am_c = c * oc_stride;

        for (Py_ssize_t oy = 0; oy < Hout; oy++) {
            Py_ssize_t iy0 = oy * s;
            Py_ssize_t y_cy = y_c + oy * Wout;
            Py_ssize_t am_cy = am_c + oy * Wout;

            Py_ssize_t x_win_row0 = x_c + iy0 * xs1;

            for (Py_ssize_t ox = 0; ox < Wout; ox++) {
                Py_ssize_t ix0 = ox * s;

                double best_val = -1000000.0; // -inf
                Py_ssize_t best_k = 0;
                Py_ssize_t k = 0;

                for (Py_ssize_t ky = 0; ky < ph; ky++) {
                    Py_ssize_t x_row = x_win_row0 + ky * xs1;
                    Py_ssize_t x_col0 = x_row + ix0 * xs2;

                    for (Py_ssize_t kx = 0; kx < pw; kx++) {
                        double v = xdata[x_col0 + kx * xs2];
                        if (v > best_val) {
                            best_val = v;
                            best_k = k;
                        }
                        k++;
                    }
                }

                ydata[y_cy + ox] = best_val;
                amdata[am_cy + ox] = (double)best_k;
            }
        }
    }

    PyBuffer_Release(&yb);
    PyBuffer_Release(&amb);
    PyBuffer_Release(&xb);

    // return (y_arr, am_arr)
    PyObject *ret = PyTuple_New(2);
    PyTuple_SET_ITEM(ret, 0, y_arr);
    PyTuple_SET_ITEM(ret, 1, am_arr);
    return ret;

fail:
    PyBuffer_Release(&xb);
    return NULL;
}

static PyObject* pool_forward_avg(PyObject *self, PyObject *args) {
    Py_buffer xb = {0};
    Py_ssize_t x_off, C, H, W, xs0, xs1, xs2;
    Py_ssize_t ph, pw, s;

    if (!PyArg_ParseTuple(
            args,
            "y*nnnnnnn"
            "nnn",
            &xb, &x_off, &C, &H, &W, &xs0, &xs1, &xs2,
            &ph, &pw, &s
        )) return NULL;

    if (!ensure_double_buf(&xb, "x")) goto fail;

    if (ph <= 0 || pw <= 0 || s <= 0) {
        PyErr_SetString(PyExc_ValueError, "ph/pw/stride must be > 0");
        goto fail;
    }

    Py_ssize_t Hout = (H - ph) / s + 1;
    Py_ssize_t Wout = (W - pw) / s + 1;
    if (Hout <= 0 || Wout <= 0) {
        PyErr_SetString(PyExc_ValueError, "Invalid output shape for pooling");
        goto fail;
    }

    double *xdata = (double*)xb.buf;

    Py_ssize_t out_n = C * Hout * Wout;
    PyObject *y_arr = make_zeroed_array_d(out_n);
    if (!y_arr) goto fail;

    Py_buffer yb = {0};
    if (PyObject_GetBuffer(y_arr, &yb, PyBUF_WRITABLE) != 0) { Py_DECREF(y_arr); goto fail; }
    double *ydata = (double*)yb.buf;

    double inv = 1.0 / (double)(ph * pw);
    Py_ssize_t oc_stride = Hout * Wout;

    for (Py_ssize_t c = 0; c < C; c++) {
        Py_ssize_t x_c = x_off + c * xs0;
        Py_ssize_t y_c = c * oc_stride;

        for (Py_ssize_t oy = 0; oy < Hout; oy++) {
            Py_ssize_t iy0 = oy * s;
            Py_ssize_t y_cy = y_c + oy * Wout;

            Py_ssize_t x_win_row0 = x_c + iy0 * xs1;

            for (Py_ssize_t ox = 0; ox < Wout; ox++) {
                Py_ssize_t ix0 = ox * s;

                double acc = 0.0;
                for (Py_ssize_t ky = 0; ky < ph; ky++) {
                    Py_ssize_t x_row = x_win_row0 + ky * xs1;
                    Py_ssize_t x_col0 = x_row + ix0 * xs2;
                    for (Py_ssize_t kx = 0; kx < pw; kx++) {
                        acc += xdata[x_col0 + kx * xs2];
                    }
                }

                ydata[y_cy + ox] = acc * inv;
            }
        }
    }

    PyBuffer_Release(&yb);
    PyBuffer_Release(&xb);
    return y_arr;

fail:
    PyBuffer_Release(&xb);
    return NULL;
}

static PyObject* pool_backward_max(PyObject *self, PyObject *args) {
    Py_buffer dyb = {0}, amb = {0};

    // dy args3
    Py_ssize_t dy_off, dyC, Hout, Wout, dys0, dys1, dys2;

    // argmax args3
    Py_ssize_t am_off, amC, amH, amW, ams0, ams1, ams2;

    // extras
    Py_ssize_t H, W, ph, pw, s;

    if (!PyArg_ParseTuple(
            args,
            "y*nnnnnnn"  // dy: buf, off, C, Hout, Wout, st0, st1, st2
            "y*nnnnnnn"  // am: buf, off, C, Hout, Wout, st0, st1, st2
            "nnnnn",     // H, W, ph, pw, s
            &dyb, &dy_off, &dyC, &Hout, &Wout, &dys0, &dys1, &dys2,
            &amb, &am_off, &amC, &amH, &amW, &ams0, &ams1, &ams2,
            &H, &W, &ph, &pw, &s
        )) {
        return NULL;
    }

    if (!ensure_double_buf(&dyb, "dy") || !ensure_double_buf(&amb, "argmax")) goto fail;

    // Debug (optional): uncomment to verify what C received
    // PySys_WriteStdout("C parsed: dyC=%zd Hout=%zd Wout=%zd | amC=%zd amH=%zd amW=%zd\n",
    //                   dyC, Hout, Wout, amC, amH, amW);

    if (amC != dyC || amH != Hout || amW != Wout) {
        PyErr_SetString(PyExc_ValueError, "argmax shape mismatch vs dy");
        goto fail;
    }

    if (ph <= 0 || pw <= 0 || s <= 0) {
        PyErr_SetString(PyExc_ValueError, "ph/pw/stride must be > 0");
        goto fail;
    }

    // Validate dy shape corresponds to (H,W,ph,pw,s)
    if (Hout != (H - ph) / s + 1 || Wout != (W - pw) / s + 1) {
        PyErr_SetString(PyExc_ValueError, "dy shape does not match pooling output shape from (H,W,ph,pw,s)");
        goto fail;
    }

    double *dydata = (double*)dyb.buf;
    double *amdata = (double*)amb.buf;

    Py_ssize_t dx_n = dyC * H * W;
    PyObject *dx_arr = make_zeroed_array_d(dx_n);
    if (!dx_arr) goto fail;

    Py_buffer dxb = {0};
    if (PyObject_GetBuffer(dx_arr, &dxb, PyBUF_WRITABLE) != 0) { Py_DECREF(dx_arr); goto fail; }
    double *dxdata = (double*)dxb.buf;

    // contiguous dx: (C,H,W)
    Py_ssize_t dc_stride = H * W;

    for (Py_ssize_t c = 0; c < dyC; c++) {
        Py_ssize_t dy_c = dy_off + c * dys0;
        Py_ssize_t am_c = am_off + c * ams0;
        Py_ssize_t dx_c = c * dc_stride;

        for (Py_ssize_t oy = 0; oy < Hout; oy++) {
            Py_ssize_t iy0 = oy * s;
            Py_ssize_t dy_cy = dy_c + oy * dys1;
            Py_ssize_t am_cy = am_c + oy * ams1;

            for (Py_ssize_t ox = 0; ox < Wout; ox++) {
                double g = dydata[dy_cy + ox * dys2];

                Py_ssize_t k = (Py_ssize_t)amdata[am_cy + ox * ams2];
                Py_ssize_t ky = k / pw;
                Py_ssize_t kx = k - ky * pw;

                Py_ssize_t iy = iy0 + ky;
                Py_ssize_t ix = ox * s + kx;

                dxdata[dx_c + iy * W + ix] += g;
            }
        }
    }

    PyBuffer_Release(&dxb);
    PyBuffer_Release(&dyb);
    PyBuffer_Release(&amb);
    return dx_arr;

fail:
    PyBuffer_Release(&dyb);
    PyBuffer_Release(&amb);
    return NULL;
}

static PyObject* pool_backward_avg(PyObject *self, PyObject *args) {
    Py_buffer dyb = {0};
    Py_ssize_t dy_off, C, Hout, Wout, dys0, dys1, dys2;
    Py_ssize_t H, W, ph, pw, s;

    if (!PyArg_ParseTuple(
            args,
            "y*nnnnnnn"
            "nnnnn",
            &dyb, &dy_off, &C, &Hout, &Wout, &dys0, &dys1, &dys2,
            &H, &W, &ph, &pw, &s
        )) return NULL;

    if (!ensure_double_buf(&dyb, "dy")) goto fail;

    if (ph <= 0 || pw <= 0 || s <= 0) {
        PyErr_SetString(PyExc_ValueError, "ph/pw/stride must be > 0");
        goto fail;
    }
    if (Hout != (H - ph) / s + 1 || Wout != (W - pw) / s + 1) {
        PyErr_SetString(PyExc_ValueError, "dy shape does not match pooling output shape from (C,H,W,ph,pw,s)");
        goto fail;
    }

    double *dydata = (double*)dyb.buf;

    Py_ssize_t dx_n = C * H * W;
    PyObject *dx_arr = make_zeroed_array_d(dx_n);
    if (!dx_arr) goto fail;

    Py_buffer dxb = {0};
    if (PyObject_GetBuffer(dx_arr, &dxb, PyBUF_WRITABLE) != 0) { Py_DECREF(dx_arr); goto fail; }
    double *dxdata = (double*)dxb.buf;

    double scale = 1.0 / (double)(ph * pw);
    Py_ssize_t dc_stride = H * W;

    for (Py_ssize_t c = 0; c < C; c++) {
        Py_ssize_t dy_c = dy_off + c * dys0;
        Py_ssize_t dx_c = c * dc_stride;

        for (Py_ssize_t oy = 0; oy < Hout; oy++) {
            Py_ssize_t iy0 = oy * s;
            Py_ssize_t dy_cy = dy_c + oy * dys1;

            for (Py_ssize_t ox = 0; ox < Wout; ox++) {
                double g = dydata[dy_cy + ox * dys2] * scale;
                Py_ssize_t ix0 = ox * s;

                for (Py_ssize_t ky = 0; ky < ph; ky++) {
                    Py_ssize_t iy = iy0 + ky;
                    Py_ssize_t dx_row = dx_c + iy * W;
                    for (Py_ssize_t kx = 0; kx < pw; kx++) {
                        Py_ssize_t ix = ix0 + kx;
                        dxdata[dx_row + ix] += g;
                    }
                }
            }
        }
    }

    PyBuffer_Release(&dxb);
    PyBuffer_Release(&dyb);
    return dx_arr;

fail:
    PyBuffer_Release(&dyb);
    return NULL;
}

static PyMethodDef Methods[] = {
    {"convolve_forward", convolve_forward, METH_VARARGS, "forward prop of conv layer"},
    {"convolve_backward", convolve_backward, METH_VARARGS, "back prop of conv layer"},
    {"dense_forward",  dense_forward,  METH_VARARGS, "forward prop of dense layer"},
    {"dense_backward", dense_backward, METH_VARARGS, "back prop of dense layer"},
    {"pool_forward_max",  pool_forward_max,  METH_VARARGS, "max forward prop of pool layer"},
    {"pool_forward_avg",  pool_forward_avg,  METH_VARARGS, "avg forward prop of pool layer"},
    {"pool_backward_max", pool_backward_max, METH_VARARGS, "max back prop of pool layer"},
    {"pool_backward_avg", pool_backward_avg, METH_VARARGS, "avg back prop of pool layer"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "cnn", NULL, -1, Methods
};

PyMODINIT_FUNC PyInit_cnn(void) {
    return PyModule_Create(&moduledef);
}