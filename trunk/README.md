Port to Dart of NeuQuant algorithm by Anthony Dekker, 1994:  
<http://members.ozemail.com.au/~dekker/NEUQUANT.HTML>

### Example ###

To the left a 32 bits RGBA image with 68,289 unique colors. To the right the same image with just 256 colors after quantization:

![Neuquant](http://www.inmensia.com/files/pictures/posts/neuquant_fallas_01.jpg)

### Usage ###

The algorithm performs quantization of a 32-bit RGBA image to 8-bit colour. By adjusting a sampling factor, the algorithm can either produce extremely high-quality images slowly, or produce good images in reasonable times. A sampling factor of 10 gives a substantial speed-up, with a small quality penalty.

```
import "dart:html";
import "package:neuquant/neuquant.dart";

void main() {
  var image = query("#image") as ImageElement;
  var canvas = query("#canvas") as CanvasElement;
  var context = canvas.context2D;

  context.drawImage(image, 0, 0);

  var imageData = context.getImageData(0, 0, canvas.width, canvas.height);

  neuquant(imageData.data, 10);

  context.putImageData(imageData, 0, 0);
}
```