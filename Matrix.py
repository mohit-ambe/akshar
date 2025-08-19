from c_extension import mtrx


class Matrix():
    """
    A Matrix class that can perform addition and multiplication operations.
    """

    def __init__(self, *args):
        """
        Initialize a Matrix object.

        No arguments passed creates an empty Matrix.

        Passing r (int) and c (int) creates an r rows by c columns Matrix of zeroes.

        Passing mtrx (list[list[int]]) creates a Matrix with its specified values.
        """

        self.data = list()
        self.rows, self.cols = 0, 0
        self.floating_digits = 2

        self.row = lambda ri:self.data[ri]
        self.col = lambda ci:list(map(lambda d:d[ci], self.data))
        self.dot = lambda dr, dc:sum([dr[i] * dc[i] for i in range(len(dr))])

        if not args:
            return

        if type(args[0]) is list:
            mtrx = args[0].copy()
            if not all(len(row) == len(mtrx[0]) for row in mtrx):
                raise ValueError("Unable to parse matrix of irregular shape")
            self.rows, self.cols = len(mtrx), len(mtrx[0])
            for mr in range(self.rows):
                self.data.append(list())
                for mc in range(self.cols):
                    x = mtrx[mr][mc]
                    if type(x) not in (int, float):
                        raise TypeError("Matrix values must of type int/float")
                    self.data[mr].append(x)
        elif len(args) == 2:
            r, c = args
            if type(r) is int and type(c) is int and r > 0 and c > 0:
                self.data = [[0 for _ in range(c)] for _ in range(r)]
                self.rows, self.cols = r, c
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
        m1 = [_ for row in self.data for _ in row]
        m2 = [_ for row in addend.data for _ in row]
        matrix_sum = mtrx.add(m1, m2)
        matrix_sum = [matrix_sum[i:i + self.cols] for i in range(0, len(matrix_sum), self.cols)]
        return Matrix(matrix_sum)

    def __mul__(self, multi: (float, "Matrix")) -> "Matrix":
        """
        Multiplies two matrices, returning the product of

        self x multi using an extension library in C.

        Or multiply a matrix with a constant, returning

        M[i][j] x multi for all i, j
        """

        if type(multi) not in [Matrix, float, int]:
            raise TypeError(f"unable to perform * for types Matrix and {type(multi)}")

        if type(multi) in [float, int]:
            m = [_ for row in self.data for _ in row]
            product = mtrx.scalar_multiply(m, multi)
            product = [product[i:i + self.cols] for i in range(0, len(product), self.cols)]
            return Matrix(product)
        elif type(multi) is Matrix:
            m1 = [_ for row in self.data for _ in row]
            m2 = [_ for row in multi.data for _ in row]
            product = mtrx.matrix_multiply(self.rows, self.cols, multi.rows, multi.cols, m1, m2)
            product = [product[i:i + multi.cols] for i in range(0, len(product), multi.cols)]
            return Matrix(product)

    def __sub__(self, subtrahend: "Matrix") -> "Matrix":
        """
        Subtracts two matrices using an extension library in C.
        """
        return self.__add__(subtrahend.__mul__(-1))

    def apply(self, func) -> "Matrix":
        """
        Applies a function to all values
        """
        return Matrix([list(map(func, row)) for row in self.data])

    def transpose(self) -> "Matrix":
        """
        Transpose the matrix using an extension library in C
        """
        m = [_ for row in self.data for _ in row]
        product = mtrx.transpose(self.rows, self.cols, m)
        product = [product[i:i + self.rows] for i in range(0, len(product), self.rows)]
        return Matrix(product)

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
                s += f"{round(self.data[r][c], self.floating_digits):.{self.floating_digits}f}\t"
            string += f"{s.strip()}\n"

        return string if string else "<empty matrix>"
