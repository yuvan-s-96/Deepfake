import React from 'react';
import { Carousel } from 'primereact/carousel';

const images = [
  { src: 'https://d12aarmt01l54a.cloudfront.net/cms/images/Media-20191212140232/808-440.png', alt: 'Image 1' },
  { src: 'https://www.mdpi.com/sensors/sensors-21-07367/article_deploy/html/images/sensors-21-07367-g001.png', alt: 'Image 2' },
  { src: 'image3.jpg', alt: 'Image 3' },
  // Add more image objects as needed
];

export default function ImageCarousel() {
  const itemTemplate = (image) => (
    <div>
      <img src={image.src} alt={image.alt} style={{ width: '100%' }} />
    </div>
  );

  return (
    <Carousel value={images} itemTemplate={itemTemplate} />
  );
}
