# STEADY
**S**tock **T**racking and **E**valuation **A**lgorithm for **D**ynamic **Y**ields

# Algorithm

```
f(
    Close Price,
    Open Price,
    Lowest Price,
    Highest Price,
    NTransactions,
    Trading Volume
) => {Close of next Days}
```

```
┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐
│  123  │   │  234  │   │  345  │   │  456  │   │  567  │   │  678  │
└───────┘   └───────┘   └───────┘   └───────┘   └───────┘   └───────┘
                              ╔═══════╗
                              ║       ║
                              ║       ║
                              ║       ║
                              ║       ║
                              ║       ║
                              ║       ║
                              ╚═══════╝
                               ↓↓↓↓↓↓↓
                    ┌───────┐ ┌───────┐ ┌───────┐
                    │  101  │ │  202  │ │  303  │
                    └───────┘ └───────┘ └───────┘
```
# Results
![I1](/assets/I1.jpeg)
![I2](/assets/I2.jpeg)
![I3](/assets/I3.jpeg)
![I4](/assets/I4.jpeg)
![I5](/assets/I5.jpeg)
![I6](/assets/I6.jpeg)

# License

The MIT License (MIT)

Copyright (c) 2024 Arman Behravan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
