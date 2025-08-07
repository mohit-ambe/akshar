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

    def __add__(self, addend: list[list[int]]) -> "Matrix":
        """
        Adds two matrices.
        """

        if type(addend) is not Matrix:
            raise TypeError(f"unable to perform + for types Matrix and {type(addend)}")
        if self.rows != addend.rows or self.cols != addend.cols:
            raise ValueError(
                f"Shape Mismatch: expected {(self.rows, self.cols)}, received {(addend.rows, addend.cols)}")

        sum = Matrix(self.rows, self.cols)
        for ar in range(self.rows):
            for ac in range(self.cols):
                x = addend.data[ar][ac]
                if type(x) not in (int, float):
                    raise TypeError("Matrix values must of type int/float")
                sum.data[ar][ac] = self.data[ar][ac] + x

        return sum

    def __sub__(self, subtrahend):
        return self.__add__(subtrahend * -1)

    def __mul__(self, multi) -> "Matrix":
        """
        Multiplies two matrices, returning the product of

        self x multi.
        """

        if type(multi) not in [Matrix, int]:
            raise TypeError(f"unable to perform * for types Matrix and {type(multi)}")

        if type(multi) is int:
            product = Matrix(self.rows, self.cols)
            for mr in range(self.rows):
                for mc in range(self.cols):
                    product[mr][mc] = self.data[mr][mc] * multi
            return product

        if self.cols != multi.rows:
            raise ValueError(f"Shape Mismatch: expected {self.cols} rows, received {multi.rows}")
        product = Matrix(self.rows, multi.cols)
        for mr in range(self.rows):
            for mc in range(multi.cols):
                try:
                    product[mr][mc] = self.dot(self.row(mr), multi.col(mc))
                except TypeError:
                    raise TypeError("Matrix values must of type int/float")

        return product

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
