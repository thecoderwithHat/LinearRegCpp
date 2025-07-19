# Linear Regression in C++ 📈

*Because I was bored and thought "Hey, let's reinvent the wheel... but in C++!"*

## What is this?

A simple linear regression implementation written in C++ during one of those weekend afternoons when you've already watched everything on Netflix and scrolled through social media twice. You know the feeling.

## Why C++?

Good question. Python exists. R exists. Even Excel can do this. But here we are, manually managing memory and wrestling with pointers because apparently that's how I choose to spend my free time.

## What it does

- Takes in data points (x, y coordinates)
- Fits a line through them using the least squares method
- Gives you the equation: `y = mx + b`
- Makes you feel like a data scientist (sort of)

## Features

- ✅ Basic linear regression algorithm
- ✅ Calculates slope (m) and y-intercept (b)
- ✅ Computes R-squared value for goodness of fit
- ✅ Handles edge cases (because segfaults aren't fun)
- ✅ Memory management that won't make your computer cry
- ❌ GUI (this is C++, not a miracle)



## Dependencies

- A C++ compiler that supports C++17 (because we're not animals)
- Standard library (vector, iostream, cmath)
- Your patience while debugging pointer arithmetic

## Known Issues

- Doesn't handle vertical lines (infinite slope makes computers sad)
- Assumes your data actually has a linear relationship (optimistic, I know)
- No built-in data visualization (use Python for that, trust me and get a life and WTF is wrong with you?)

## Future Improvements (aka "Things I'll probably never do")

- [ ] Multiple linear regression
- [ ] Polynomial regression
- [ ] Ridge/Lasso regression
- [ ] A GUI that doesn't look like it's from 1995
- [ ] GPU acceleration (because why not overcomplicate things?)

## Contributing

Found a bug? Have a suggestion? Want to add more features to this monument of boredom-driven development? 

Feel free to open an issue or submit a pull request. Just remember: this started as a "I'm bored" project, so let's keep the scope reasonable.


## Acknowledgments

- My boredom, for being the driving force behind this project
- Stack Overflow(Yes!, I still use it), for helping me remember C++ syntax
- Coffee, for making late-night coding sessions possible
- That one book chapter that explained linear regression really well

---

*"Sometimes the best projects come from the most random moments of boredom."* - Me, trying to justify spending a weekend on this

**Disclaimer**: This is a learning project. For production use, please consider established libraries like Eigen, MLPack, or just use Python like a normal person. And the Readme is AI generated beacuse I was feeling LAZY!!!