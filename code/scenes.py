from enum import Enum
import os

class Scene(Enum):
    BEDROOM = "bedroom", "Bedroom", 1
    COAST = "Coast", "Coast", 2
    FOREST = "Forest", "Forest", 3
    HIGHWAY = "Highway", "Highway", 4
    INDUSTRIAL = "industrial", "Industrial", 5
    INSIDE_CITY = "Insidecity", "Inside city", 6
    KITCHEN = "kitchen", "Kitchen", 7
    LIVING_ROOM = "livingroom", "Living room", 8
    MOUNTAIN = "Mountain", "Mountain", 9
    OFFICE = "Office", "Office", 10
    OPEN_COUNTRY = "OpenCountry", "Open country", 11
    STORE = "store", "Store", 12
    STREET = "Street", "Street", 13
    SUBURB = "Suburb", "Suburb", 14
    TALL_BUILDING = "TallBuilding", "Tall building", 15

    def __new__(cls, directory, title, index):
        obj = object.__new__(cls)
        obj._value_ = directory
        obj.directory = directory
        obj.title = title
        obj.index = index
        obj.out = directory.lower()
        return obj

    @classmethod
    def from_index(cls, index:int):
        for scene in cls:
            if scene.index == index:
                return scene
        raise ValueError(f"No Scene found for index {index}")

    @classmethod
    def from_path(cls, path:str):
        return Scene(path.split(os.sep)[-2])

    @classmethod
    def from_directory(cls, directory:str):
        return Scene(directory)

