from pathlib import Path
from typing import List, Dict
import json
from data import Math23kProblem

if __name__ == '__main__':
    single_problem = Math23kProblem(
        oldText="The children in the second grade of Zhenhai Yale School went to the side of a small road to plant trees. The children planted a tree every 2 meters (trees were planted at both ends of the road), and finally found that a total of 11 trees were planted. How many meters is the path long?",
        oldFormula=["x=(11-1)*2"],
        oldAnswer=["20"],
        mwp_template="The children in the second grade of Zhenhai Yale School went to the side of a small road to plant trees . The children planted a tree every num3 meters ( trees were planted at both ends of the road ) , and finally found that a total of num1 trees were planted . How many meters is the path long ?",
        eqs_template=["x = ( num1 - num2 ) * num3"]
    )
    print(single_problem.as_dict())