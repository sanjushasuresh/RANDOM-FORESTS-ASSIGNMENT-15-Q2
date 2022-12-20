About the data: <br>
Let’s consider a Company dataset with around 10 variables and 400 records. <br>
The attributes are as follows: <br>
Sales -- Unit sales (in thousands) at each location<br>
Competitor Price -- Price charged by competitor at each location<br>
Income -- Community income level (in thousands of dollars)<br>
Advertising -- Local advertising budget for company at each location (in thousands of dollars)<br>
Population -- Population size in region (in thousands)<br>
Price -- Price company charges for car seats at each site<br>
Shelf Location at stores -- A factor with levels Bad, Good and Medium indicating the quality of the shelving location for the car seats at each site<br>
Age -- Average age of the local population<br>
Education -- Education level at each location<br>
Urban -- A factor with levels No and Yes to indicate whether the store is in an urban or rural location<br>
US -- A factor with levels No and Yes to indicate whether the store is in the US or not<br>

The company dataset looks like this: <br>
Problem Statement:<br>
A cloth manufacturing company is interested to know about the segment or attributes causes high sale. <br>
Approach - A Random Forest can be built with the target variable Sales (we will first convert it in categorical variable) & all other variable will be independent in the analysis.
