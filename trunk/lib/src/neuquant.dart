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

/* NeuQuant Neural-Net Quantization Algorithm
 * ------------------------------------------
 *
 * Copyright (c) 1994 Anthony Dekker
 *
 * NEUQUANT Neural-Net quantization algorithm by Anthony Dekker, 1994.
 * See "Kohonen neural networks for optimal colour quantization"
 * in "Network: Computation in Neural Systems" Vol. 5 (1994) pp 351-367.
 * for a discussion of the algorithm.
 * See also  http://members.ozemail.com.au/~dekker/NEUQUANT.HTML
 *
 * Any party obtaining a copy of these files from the author, directly or
 * indirectly, is granted, free of charge, a full and unrestricted irrevocable,
 * world-wide, paid up, royalty-free, nonexclusive right and license to deal
 * in this software and documentation files (the "Software"), including without
 * limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons who receive
 * copies from any such party to do so, with the only requirement being
 * that this copyright notice remain intact.
 */

part of neuquant;

// Number of colours used.
// For 256 colours, fixed arrays need 8kb, plus space for the image.
const int _NET_SIZE = 256;

// Four primes near 500 - assume no image has a length so large
// that it is divisible by all four primes.
const int _PRIME_1 = 499;
const int _PRIME_2 = 491;
const int _PRIME_3 = 487;
const int _PRIME_4 = 503;

// Minimum size for input image.
const int _MIN_PICTURE_BYTES = 4 * _PRIME_4;

// Network Definitions
// -------------------
const int _MAX_NET_POS       = _NET_SIZE - 1;
const int _NET_BIAS_SHIFT    = 4;   // Bias for colour values.
const int _N_CYCLES          = 100; // Number of learning cycles.

// Definitions for freq and bias.
const int _INT_BIAS_SHIFT    = 16;  // Bias for fractions.
const int _INT_BIAS          = 1 << _INT_BIAS_SHIFT;
const int _GAMMA_SHIFT       = 10;                       // gamma = 1024
const int _GAMMA             = 1 << _GAMMA_SHIFT;
const int _BETA_SHIFT        = 10;
const int _BETA              = _INT_BIAS >> _BETA_SHIFT; // beta = 1/1024
const int _BETA_GAMMA        = _INT_BIAS << (_GAMMA_SHIFT - _BETA_SHIFT);

// Definitions for decreasing radius factor.
const int _INIT_RAD          = _NET_SIZE >> 3; // For 256 colors, radius starts
const int _RADIUS_BIAS_SHIFT = 6;              // at 32.0 biased by 6 bits
const int _RADIUS_BIAS       = 1 << _RADIUS_BIAS_SHIFT;
const int _INIT_RADIUS       = _INIT_RAD * _RADIUS_BIAS; // and decreases by a
const int _RADIUS_DEC        = 30;             // factor of 1/30 each cycle.

// Definitions for decreasing alpha factor.
const int _ALPHA_BIAS_SHIFT  = 10;  // alpha starts at 1.0
const int _INIT_ALPHA        = 1 << _ALPHA_BIAS_SHIFT;
int _alphadec;                      // biased by 10 bits.

// radbias and alpharadbias used for radpower calculation.
const int _RAD_BIAS_SHIFT    = 8;
const int _RAD_BIAS          = 1 << _RAD_BIAS_SHIFT;
const int _ALPHA_RAD_BSHIFT  = _ALPHA_BIAS_SHIFT + _RAD_BIAS_SHIFT;
const int _ALPHA_RAD_BIAS    = 1 << _ALPHA_RAD_BSHIFT;

// Types and Global Variables
// --------------------------
List<int> _thepicture;  // The input image itself.
int _lengthcount;       // lengthcount = Height * Width * 4
int _samplefac;         // Sampling factor 1..30

//typedef int pixel[4];               // RGBc
final List<int> _network = new List<int>(_NET_SIZE * 4);  // The network itself.

final List<int> _netindex = new List<int>(256); // For network lookup - really 256.

final List<int> _bias = new List<int>(_NET_SIZE);  // bias and freq arrays for learning.
final List<int> _freq = new List<int>(_NET_SIZE);
final List<int> _radpower = new List<int>(_INIT_RAD); // radpower for precomputation.

/// Initialise network in range (0,0,0) to (255,255,255) and set parameters.
void _initnet(List<int> thepic, int sample) {
  _thepicture = thepic;
  _lengthcount = thepic.length;
  _samplefac = sample;

  for (var i = 0, p = 0; i < _NET_SIZE; ++ i, p += 4) {
    _network[p] = _network[p + 1] = _network[p + 2] =
        (i << (_NET_BIAS_SHIFT + 8)) ~/ _NET_SIZE;
    _freq[i] = _INT_BIAS ~/ _NET_SIZE;  // 1 / netsize
    _bias[i] = 0;
  }
}

/// Unbias network to give byte values 0..255 and record position i to prepare for sort.
void _unbiasnet() {
  for (var i = 0, p = 0; i < _NET_SIZE; ++ i, p += 4) {
    for (var j = 0; j < 3; ++ j) {
      var temp = (_network[p + j] + (1 << (_NET_BIAS_SHIFT - 1))) >> _NET_BIAS_SHIFT;
      if (temp > 255) {
        temp = 255;
      }
      _network[p + j] = temp;
    }
    _network[p + 3] = i;      // Record colour number.
  }
}

/// Output colour map.
List<int> _writecolourmap() {
  var map = new List<int>(_NET_SIZE * 3);
  for (var i = 0, j = 0; i < _network.length; ++ i) {
    map[j ++] = _network[i ++];
    map[j ++] = _network[i ++];
    map[j ++] = _network[i ++];
  }
  return map;
}

/// Insertion sort of network and building of netindex[0..255] (to do after unbias).
void _inxbuild() {
  var previouscol = 0;
  var startpos = 0;
  for (var i = 0, p = 0; i < _NET_SIZE; ++ i, p += 4) {
    var smallpos = i;
    var smallval = _network[p + 1];     // Index on G.
    // Find smallest in i..netsize-1
    for (var j = i + 1, q = p + 4; j < _NET_SIZE; ++ j, q += 4) {
      if (_network[q + 1] < smallval) { // Index on G.
        smallpos = j;
        smallval = _network[q + 1];     // Index on G.
      }
    }
    // Swap p (i) and q (smallpos) entries.
    if (i != smallpos) {
      var q = smallpos << 2;
      var t = _network[q]; _network[q] = _network[p]; _network[p] = t;
      t = _network[q + 1]; _network[q + 1] = _network[p + 1]; _network[p + 1] = t;
      t = _network[q + 2]; _network[q + 2] = _network[p + 2]; _network[p + 2] = t;
      t = _network[q + 3]; _network[q + 3] = _network[p + 3]; _network[p + 3] = t;
    }
    // smallval entry is now in position i.
    if (smallval != previouscol) {
      _netindex[previouscol] = (startpos + i) >> 1;
      for (var j = previouscol + 1; j < smallval; ++ j) {
        _netindex[j] = i;
      }
      previouscol = smallval;
      startpos = i;
    }
  }
  _netindex[previouscol] = (startpos + _MAX_NET_POS) >> 1;
  for (var j = previouscol + 1; j < 256; ++ j) { // Really 256.
    _netindex[j] = _MAX_NET_POS;
  }
}

/// Search for RGB values 0..255 (after net is unbiased) and return colour index.
int _inxsearch(int r, int g, int b) {
  var bestd = 1000;      // Biggest possible dist is 256 * 3
  var best = -1;
  var i = _netindex[g], p = i << 2;  // Index on G.
  var j = i - 1, q = j << 2;         // Start at netindex[g] and work outwards.
  while ((i < _NET_SIZE) || (j >= 0)) {
    if (i < _NET_SIZE) {
      var dist = _network[p + 1] - g; // Inx key.
      if (dist >= bestd) {
        i = _NET_SIZE; // Stop iteration.
      } else {
        ++ i;
        if (dist < 0) {
          dist = -dist;
        }
        var a = _network[p + 2] - b;
        if (a < 0) {
          a = -a;
        }
        dist += a;
        if (dist < bestd) {
          a = _network[p] - r;
          if (a < 0) {
            a = -a;
          }
          dist += a;
          if (dist < bestd) {
            bestd = dist;
            best = _network[p + 3];
          }
        }
        p += 4;
      }
    }
    if (j >= 0) {
      var dist = g - _network[q + 1]; // Inx key - reverse dif.
      if (dist >= bestd) {
        j = -1; // Stop iteration.
      } else {
        -- j;
        if (dist < 0) {
          dist = -dist;
        }
        var a = _network[q + 2] - b;
        if (a < 0) {
          a = -a;
        }
        dist += a;
        if (dist < bestd) {
          a = _network[q] - r;
          if (a < 0) {
            a = -a;
          }
          dist += a;
          if (dist < bestd) {
            bestd = dist;
            best = _network[q + 3];
          }
        }
        q -= 4;
      }
    }
  }
  return best;
}

/// Search for biased RGB values.
/// Finds closest neuron (min dist) and updates freq.
/// Finds best neuron (min dist - bias) and returns position.
/// For frequently chosen neurons, freq[i] is high and bias[i] is negative.
/// bias[i] = gamma * ((1 / netsize) - freq[i])
int _contest(int r, int g, int b) {
  var bestd = 0x7FFFFFFF; //~(1 << 31);
  var bestbiasd = bestd;
  var bestpos = -1;
  var bestbiaspos = bestpos;
  for (var i = 0, n = 0, p = 0, f = 0; i < _NET_SIZE; ++ i, n += 4) {
    var dist = _network[n + 2] - b;
    if (dist < 0) {
      dist = -dist;
    }
    var a = _network[n + 1] - g;
    if (a < 0) {
      a = -a;
    }
    dist += a;
    a = _network[n] - r;
    if (a < 0) {
      a = -a;
    }
    dist += a;
    if (dist < bestd) {
      bestd = dist;
      bestpos = i;
    }

    //jcmellado: Changed to get the same result in the JavaScript version.
    //Old code: var biasdist = dist - (_bias[p] >> (_INT_BIAS_SHIFT - _NET_BIAS_SHIFT));
    var biasdist = dist - (_bias[p] ~/ 4096);
    if (biasdist < bestbiasd) {
      bestbiasd = biasdist;
      bestbiaspos = i;
    }
    var betafreq = _freq[f] >> _BETA_SHIFT;
    _freq[f++] -= betafreq;
    _bias[p++] += betafreq << _GAMMA_SHIFT;
  }
  _freq[bestpos] += _BETA;
  _bias[bestpos] -= _BETA_GAMMA;
  return bestbiaspos;
}

/// Move neuron i towards biased (r,g,b) by factor alpha.
void _altersingle(int alpha, int i, int r, int g, int b) {
  var n = i << 2; // Alter hit neuron.
  _network[n] -= (alpha * (_network[n] - r)) ~/ _INIT_ALPHA;
  _network[n + 1] -= (alpha * (_network[n + 1] - g)) ~/ _INIT_ALPHA;
  _network[n + 2] -= (alpha * (_network[n + 2] - b)) ~/ _INIT_ALPHA;
}

/// Move adjacent neurons by precomputed alpha*(1-((i-j)^2/[r]^2)) in radpower[|i-j|].
void _alterneigh(int rad, int i, int r, int g, int b) {
  var lo = i - rad;
  if (lo < -1) {
    lo = -1;
  }
  var hi = i + rad;
  if (hi > _NET_SIZE) {
    hi = _NET_SIZE;
  }
  var j = i + 1, n = j << 2;
  var k = i - 1, m = k << 2;
  var q = 0;
  while ((j < hi) || (k > lo)) {
    var a = _radpower[++ q];
    if (j < hi) {
      _network[n] -= (a * (_network[n] - r)) ~/ _ALPHA_RAD_BIAS;
      _network[n + 1] -= (a * (_network[n + 1] - g)) ~/ _ALPHA_RAD_BIAS;
      _network[n + 2] -= (a * (_network[n + 2] - b)) ~/ _ALPHA_RAD_BIAS;
      ++ j;
      n += 4;
    }
    if (k > lo) {
      _network[m] -= (a * (_network[m] - r)) ~/ _ALPHA_RAD_BIAS;
      _network[m + 1] -= (a * (_network[m + 1] - g)) ~/ _ALPHA_RAD_BIAS;
      _network[m + 2] -= (a * (_network[m + 2] - b)) ~/ _ALPHA_RAD_BIAS;
      -- k;
      m -= 4;
    }
  }
}

/// Main Learning Loop.
void _learn() {
  _alphadec = 30 + ((_samplefac - 1) ~/ 3);

  var p = 0;
  var samplepixels = _lengthcount ~/ (4 * _samplefac);
  var delta = samplepixels ~/ _N_CYCLES;
  var alpha = _INIT_ALPHA;
  var radius = _INIT_RADIUS;

  var rad = radius >> _RADIUS_BIAS_SHIFT;
  if (rad <= 1) {
    rad = 0;
  }
  for (var i = 0; i < rad; ++ i) {
    _radpower[i] = alpha * ((((rad * rad) - (i * i)) * _RAD_BIAS) ~/ (rad * rad));
  }

  var step;
  if ((_lengthcount % _PRIME_1) != 0) {
    step = 4 * _PRIME_1;
  } else {
    if ((_lengthcount % _PRIME_2) != 0) {
      step = 4 * _PRIME_2;
    } else {
      if ((_lengthcount % _PRIME_3) != 0) {
        step = 4 * _PRIME_3;
      } else {
        step = 4 * _PRIME_4;
      }
    }
  }

  var i = 0;
  while (i < samplepixels) {
    var r = _thepicture[p] << _NET_BIAS_SHIFT;
    var g = _thepicture[p + 1] << _NET_BIAS_SHIFT;
    var b = _thepicture[p + 2] << _NET_BIAS_SHIFT;
    var j = _contest(r, g, b);

    _altersingle(alpha, j, r, g, b);
    if (rad != 0) {
      _alterneigh(rad, j, r, g, b);   // Alter neighbours.
    }

    p += step;
    if (p >= _lengthcount) {
      p -= _lengthcount;
    }

    ++ i;
    if ((i % delta) == 0) {
      alpha -= alpha ~/ _alphadec;
      radius -= radius ~/ _RADIUS_DEC;
      rad = radius >> _RADIUS_BIAS_SHIFT;
      if (rad <= 1) {
        rad = 0;
      }
      for (var j = 0; j < rad; ++ j) {
        _radpower[j] = alpha * (((rad * rad - j * j) * _RAD_BIAS) ~/ (rad * rad));
      }
    }
  }
}
