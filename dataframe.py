from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Iterable, Sequence

# ----------------------------------------------------------------------
# A simple data type class hierarchy for illustration


class DataType(ABC):
    """
    A metadata object representing the logical value type of a cell in a data
    frame column. This metadata does not guarantee an specific underlying data
    representation
    """
    def __eq__(self, other: 'DataType'):
        return self.equals(other)

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return str(self)

    @abstractmethod
    def to_string(self) -> str:
        """
        Return human-readable representation of the data type
        """

    @abstractmethod
    def equals(self, other: 'DataType') -> bool:
        """
        Return true if other DataType contains the same metadata as this
        DataType
        """
        pass


class PrimitiveType(DataType):

    def equals(self, other: DataType) -> bool:
        return type(self) == type(other)


class NullType(PrimitiveType):
    """
    A data type whose values are always null
    """
    def to_string(self):
        return "null"


class Boolean(PrimitiveType):

    def to_string(self):
        return "bool"


class NumberType(PrimitiveType):
    pass


class IntegerType(NumberType):
    pass


class SignedIntegerType(IntegerType):
    pass


class Int8(SignedIntegerType):

    def to_string(self):
        return "int8"


class Int16(SignedIntegerType):

    def to_string(self):
        return "int16"


class Int32(SignedIntegerType):

    def to_string(self):
        return "int32"


class Int64(SignedIntegerType):

    def to_string(self):
        return "int64"


class Binary(PrimitiveType):
    """
    A variable-size binary (bytes) value
    """
    def to_string(self):
        return "binary"


class String(PrimitiveType):
    """
    A UTF8-encoded string value
    """
    def to_string(self):
        return "string"


class Object(PrimitiveType):
    """
    Any PyObject value
    """
    def to_string(self):
        return "object"


class Categorical(DataType):
    """
    A categorical value is an ordinal (integer) value that references a
    sequence of category values of an arbitrary data type
    """

    def __init__(self, index_type: IntegerType, category_type: DataType,
                 ordered: bool = False):
        self.index_type = index_type
        self.category_type = category_type
        self.ordered = ordered

    def equals(self, other: DataType) -> bool:
        return (isinstance(other, Categorical) and
                self.index_type == other.index_type and
                self.category_type == other.category_type and
                self.ordered == other.ordered)

    def to_string(self):
        return ("categorical(indices={}, categories={}, ordered={})"
                .format(str(self.index_type), str(self.category_type),
                        self.ordered))


# ----------------------------------------------------------------------
# Classes representing a column in a DataFrame


class Column(ABC):

    @property
    @abstractmethod
    def name(self) -> Any:
        pass

    @property
    @abstractmethod
    def type(self) -> DataType:
        """
        Return the logical type of each column cell value
        """
        pass

    @property
    def attrs(self) -> Mapping:
        """
        Metadata for this column. Default implementation returns empty dict
        """
        return {}

    def to_numpy(self):
        """
        Access column's data as a NumPy array. Recommended to return a view if
        able but not required
        """
        raise NotImplementedError("Conversion to NumPy not available")

    def to_arrow(self, **kwargs):
        """
        Access column's data in the Apache Arrow format as pyarrow.Array or
        ChunkedArray. Recommended to return a view if able but not required
        """
        raise NotImplementedError("Conversion to Arrow not available")


# ----------------------------------------------------------------------
# DataFrame: the main public API


class DataFrame(ABC):
    """
    An abstract data frame base class.

    A "data frame" represents an ordered collection of named columns. A
    column's "name" is permitted to be any Python value, but strings are
    common. Names are not required to be unique. Columns may be accessed by
    name (when the name is unique) or by position.
    """

    def __dataframe__(self):
        """
        Idempotence of data frame protocol
        """
        return self

    @property
    @abstractmethod
    def num_columns(self):
        """
        Return the number of columns in the DataFrame
        """
        pass

    @property
    @abstractmethod
    def num_rows(self):
        """
        Return the number of rows in the DataFrame (if known)
        """
        pass

    @property
    @abstractmethod
    def column_names(self) -> Iterable[Any]:
        """
        Return the column names as a materialized sequence
        """
        pass

    @property
    def row_names(self) -> Sequence[Any]:
        """
        Return the row names (if any) as a materialized sequence. It is not
        necessary to implement this method
        """
        raise NotImplementedError("This DataFrame has no row names")

    @abstractmethod
    def get_column(self, i: int) -> Column:
        """
        Return the column at the indicated position
        """
        pass

    @abstractmethod
    def get_column_by_name(self, name: Any) -> Column:
        """
        Return the column whose name is the indicated name. If the column names
        are not unique, may raise an exception.
        """
        pass

    def select_columns(self, indices: Sequence[int]):
        """
        Create a new DataFrame by selecting a subset of columns by index
        """
        raise NotImplementedError("select_columns")

    def select_columns_by_name(self, names: Sequence[Any]):
        """
        Create a new DataFrame by selecting a subset of columns by name. If the
        column names are not unique, may raise an exception.
        """
        raise NotImplementedError("select_columns_by_name")

    def to_dict_of_numpy(self):
        """
        Convert DataFrame to a dict with column names as keys and values the
        corresponding columns converted to NumPy arrays
        """
        raise NotImplementedError("TODO")
