"""
Base objects for all modules.
"""

from typing import Any, List, Optional, Type


class Corpus:
    """Descriptor that stores corpus of `Vocabulary` or similar instances.

    Parameters
    ----------
    attribute_name : str
        The attribute name of the processed instance.
    """

    __slots__ = (
        'attribute_name',
    )
    __cache = dict()

    def __init__(self, attribute_name: str):
        self.attribute_name = attribute_name

    def __get__(
            self,
            instance: Any,
            owner: Optional[Type] = None,
    ) -> List[List[int]]:
        """Obtain a corpus from instance (e.g. all tokens from `Vocabulary`).

        Returns
        -------
        corpus : list of list of int
            Original data as list of token id lists.

        Raises
        ------
        AttributeError
            If getattr(instance, self.attribute_name, None) is None.
        """
        result = self.__cache.get(id(instance))
        if result is not None:
            return result

        try:
            return self.__cache.setdefault(
                id(instance),
                [
                    instance[line]
                    for line in getattr(instance, self.attribute_name)
                ],
            )
        except AttributeError as err:
            err.args = (
                f"{self.attribute_name} of {instance!r} is empty; "
                f"see documentation of {instance!r}.",
            )
            raise

    def __set__(
            self,
            instance: Any,
            value: Any,
    ):
        """Modify the attribute of `instance`.

        Raises
        ------
        AttributeError
            The descriptor is read-only.
        """
        raise AttributeError(
            f"Cannot set attribute {self.attribute_name} of {instance!r}."
        )
