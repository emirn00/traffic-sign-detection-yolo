import { Component, ElementRef, ViewChild, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { lastValueFrom } from 'rxjs';

interface Detection {
  label: string;
  confidence: number;
  bbox: number[]; // [left, top, right, bottom]
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './app.html',
  styleUrls: ['./app.css']
})
export class App {
  @ViewChild('imageCanvas') canvasRef!: ElementRef<HTMLCanvasElement>;
  
  selectedFile: File | null = null;
  selectedFileName: string = '';
  fileSelected: boolean = false;
  
  imageUrl: string | null = null;
  isProcessing: boolean = false;
  hasResult: boolean = false;
  errorMessage: string = '';
  
  detections: Detection[] = [];
  
  private apiUrl = 'http://127.0.0.1:8000/predict';

  constructor(private http: HttpClient) {}

  onFileSelected(event: any): void {
    const file = event.target.files[0];
    if (file) {
      this.selectedFile = file;
      this.selectedFileName = file.name;
      this.fileSelected = true;
      
      // Görüntüyü önizlemek için URL oluştur
      const reader = new FileReader();
      reader.onload = (e: any) => {
        this.imageUrl = e.target.result;
        this.hasResult = false;
        this.detections = [];
        this.errorMessage = '';
        
        // Canvas'a orijinal resmi çiz
        setTimeout(() => this.drawImageOnCanvas(this.imageUrl!), 50);
      };
      reader.readAsDataURL(file);
    }
  }

  async detectSigns(): Promise<void> {
    if (!this.selectedFile) return;

    this.isProcessing = true;
    this.errorMessage = '';
    this.hasResult = false;

    const formData = new FormData();
    formData.append('file', this.selectedFile);

    try {
      const response = await lastValueFrom(
        this.http.post<Detection[]>(this.apiUrl, formData)
      );
      
      this.detections = response;
      this.hasResult = true;
      
      // Görüntüyü ve bbox'ları çiz
      this.drawImageOnCanvas(this.imageUrl!, this.detections);
      
    } catch (error: any) {
      console.error('Detection error:', error);
      this.errorMessage = 'Backend API ile iletişim kurulamadı veya model yüklü değil. Sunucunun çalıştığından emin olun.';
    } finally {
      this.isProcessing = false;
    }
  }

  private drawImageOnCanvas(imgUrl: string, dets: Detection[] = []): void {
    if (!this.canvasRef) return;
    
    const canvas = this.canvasRef.nativeElement;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = new Image();
    img.onload = () => {
      // Canvas boyutunu görüntüye göre ayarla (maksimum genişlik korumasıyla)
      const maxWidth = 800;
      let width = img.width;
      let height = img.height;
      
      if (width > maxWidth) {
        const ratio = maxWidth / width;
        width = maxWidth;
        height = height * ratio;
      }
      
      canvas.width = img.width;
      canvas.height = img.height;
      
      // Resmi çiz
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      
      // CSS ile ekrana sığdır
      canvas.style.maxWidth = '100%';
      canvas.style.height = 'auto';

      // Bounding box'ları çiz
      dets.forEach(det => {
        const [x1, y1, x2, y2] = det.bbox;
        const width = x2 - x1;
        const height = y2 - y1;
        
        // Kutu
        ctx.strokeStyle = '#10b981'; // Tailwind emerald-500
        ctx.lineWidth = Math.max(3, canvas.width / 200);
        ctx.strokeRect(x1, y1, width, height);
        
        // Etiket arkaplanı
        ctx.fillStyle = '#10b981';
        const fontSize = Math.max(16, canvas.width / 40);
        ctx.font = `bold ${fontSize}px Inter, sans-serif`;
        
        const text = `${det.label} ${(det.confidence * 100).toFixed(0)}%`;
        const textWidth = ctx.measureText(text).width;
        
        ctx.fillRect(x1, y1 - fontSize - 10, textWidth + 10, fontSize + 10);
        
        // Etiket metni
        ctx.fillStyle = '#ffffff';
        ctx.fillText(text, x1 + 5, y1 - 5);
      });
    };
    img.src = imgUrl;
  }
}
