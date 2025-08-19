from c_extension import mtrx


class Matrix():
    """
    A class for Matrix operations.
    """

    def __init__(self, *args):
        """
        Initialize a Matrix object.

        No arguments passed creates an empty Matrix.

        Passing r (int) and c (int) creates an r rows by c columns Matrix of zeroes.

        Passing mtrx (list[list[int]]) creates a Matrix with its specified values.

        Passing r (int), c (int), and mtrx (list[int]) creates an r x c Matrix with values of mtrx
        """

        self.data = list()
        self.rows, self.cols = 0, 0
        self.floating_digits = 2

        self.row = lambda ri:self.data[ri:ri + self.cols]
        self.col = lambda ci:self.data[ci:len(self.data):self.cols]
        self.dot = lambda dr, dc:sum([dr[i] * dc[i] for i in range(len(dr))])

        if not args:
            return

        if type(args[0]) is list and args[0]:
            mtrx = args[0].copy()
            if not all(len(row) == len(mtrx[0]) for row in mtrx):
                raise ValueError("Unable to parse matrix of irregular shape")
            self.rows, self.cols = len(mtrx), len(mtrx[0])
            for mr in range(self.rows):
                for mc in range(self.cols):
                    x = mtrx[mr][mc]
                    if type(x) not in (int, float):
                        raise TypeError("Matrix values must of type int/float")
                    self.data.append(x)
        elif len(args) == 2:
            r, c = args
            if type(r) is int and type(c) is int and r > 0 and c > 0:
                self.data = [0 for _ in range(c) for _ in range(r)]
                self.rows, self.cols = r, c
            else:
                raise TypeError
        elif len(args) == 3 and args[2]:
            r, c, mtrx = args
            valid_dimensions = type(r) is int and type(c) is int and r > 0 and c > 0
            valid_matrix = type(mtrx) is list and len(mtrx) == r * c
            if valid_dimensions and valid_matrix:
                self.rows, self.cols = r, c
                for x in mtrx:
                    if type(x) not in (int, float):
                        raise TypeError("Matrix values must of type int/float")
                    self.data.append(x)
            else:
                raise TypeError

    def __add__(self, addend: "Matrix") -> "Matrix":
        """
        Adds two matrices using an extension library in C.
        """

        if type(addend) is not Matrix:
            raise TypeError(f"unable to perform + for types Matrix and {type(addend)}")
        if self.rows != addend.rows or self.cols != addend.cols:
            raise ValueError(
                f"Shape Mismatch: expected {(self.rows, self.cols)}, received {(addend.rows, addend.cols)}")
        matrix_sum = mtrx.add(self.data, addend.data)
        return Matrix(self.rows, self.cols, matrix_sum)

    def __mul__(self, multi: (float, "Matrix")) -> "Matrix":
        """
        Multiplies two matrices, returning the product of

        self x multi using an extension library in C.

        Or multiply a matrix with a constant, returning

        M[i][j] x multi for all i, j
        """

        if type(multi) not in [Matrix, float, int]:
            raise TypeError(f"unable to perform * for types Matrix and {type(multi)}")
        elif type(multi) in [float, int]:
            product = mtrx.scalar_multiply(self.data, multi)
            return Matrix(self.rows, self.cols, product)
        elif type(multi) is Matrix:
            if self.cols != multi.rows:
                raise ValueError(f"Shape Mismatch: expected {self.cols} rows, received {multi.rows}")
            product = mtrx.matrix_multiply(self.rows, self.cols, multi.rows, multi.cols, self.data, multi.data)
            return Matrix(self.rows, multi.cols, product)

    def __sub__(self, subtrahend: "Matrix") -> "Matrix":
        """
        Subtracts two matrices using an extension library in C.
        """
        return self.__add__(subtrahend.__mul__(-1))

    def apply(self, func) -> "Matrix":
        """
        Applies a function to all values
        """
        return Matrix(self.rows, self.cols, list(map(func, self.data)))

    def transpose(self) -> "Matrix":
        """
        Transpose the matrix using an extension library in C
        """
        product = mtrx.transpose(self.rows, self.cols, self.data)
        return Matrix(self.cols, self.rows, product)

    def __getitem__(self, key):
        return self.data.__getitem__(key)

    def __setitem__(self, key, value):
        if type(value) not in (int, float):
            raise TypeError("Matrix values must of type int/float")
        return self.data.__setitem__(key, value)

    def rounding(self, x=100):
        """
        Set how many decimal places are shown for the values' string representation.
        """
        self.floating_digits = x

    def __str__(self):
        """
        A grid style string representation for the Matrix.
        """

        string = ""

        for r in range(self.rows):
            s = ""
            for c in range(self.cols):
                s += f"{round(self.data[r * self.cols + c], self.floating_digits):.{self.floating_digits}f}\t"
            string += f"{s.strip()}\n"

        return string if string else "<empty matrix>"
