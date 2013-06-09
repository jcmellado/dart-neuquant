/*
  Copyright (c) 2013 Juan Mellado

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
*/

library neuquant;

part "src/neuquant.dart";

/// The algorithm performs quantization of a 32-bit RGBA [image] to 8-bit colour.
/// By adjusting a [samplingFactor], the algorithm can either produce extremely
/// high-quality images slowly, or produce good images in reasonable times.
/// A sampling factor of 10 gives a substantial speed-up, with a small quality penalty.
void neuquant(List<int> image, int samplingFactor) {
  _initnet(image, samplingFactor);
  _learn();
  _unbiasnet();
  var map =  _writecolourmap();
  _inxbuild();

  // Change colors.
  for (var i = 0; i < image.length; i += 4) {
    var index = _inxsearch(image[i], image[i + 1], image[i + 2]) * 3;
    image[i] = map[index];
    image[i + 1] = map[index + 1];
    image[i + 2] = map[index + 2];
  }
}
