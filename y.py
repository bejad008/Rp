import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class FreeAIBot:
    def __init__(self, telegram_token):
        self.telegram_token = telegram_token
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Pilihan model gratis dari Hugging Face
        # Anda bisa ganti dengan model lain sesuai kebutuhan
        self.model_options = {
            "microsoft/DialoGPT-large": "Conversational AI (Ringan)",
            "microsoft/DialoGPT-medium": "Conversational AI (Sedang)", 
            "facebook/blenderbot-400M-distill": "Chatbot Facebook (Ringan)",
            "microsoft/CodeBERT-base": "Code Understanding",
            "codellama/CodeLlama-7b-hf": "Code Generation (Berat)",
            "mistralai/Mistral-7B-Instruct-v0.1": "Instruction Following (Berat)"
        }
        
        # Model yang akan digunakan (pilih yang sesuai dengan spek komputer)
        self.current_model_name = "microsoft/DialoGPT-medium"  # Model default
        self.model = None
        self.tokenizer = None
        self.generator = None
        
        self.load_model()
    
    def load_model(self):
        """Load model yang dipilih"""
        try:
            print(f"Loading model: {self.current_model_name}")
            
            # Untuk model conversational
            if "DialoGPT" in self.current_model_name:
                self.tokenizer = AutoTokenizer.from_pretrained(self.current_model_name)
                self.model = AutoModelForCausalLM.from_pretrained(self.current_model_name)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    
            # Untuk model BlenderBot
            elif "blenderbot" in self.current_model_name:
                self.generator = pipeline(
                    "conversational",
                    model=self.current_model_name,
                    device=0 if torch.cuda.is_available() else -1
                )
            
            # Untuk model instruction-following
            else:
                self.generator = pipeline(
                    "text-generation",
                    model=self.current_model_name,
                    device=0 if torch.cuda.is_available() else -1,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                
            print(f"Model {self.current_model_name} loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback ke model yang lebih ringan
            self.current_model_name = "microsoft/DialoGPT-medium"
            self.tokenizer = AutoTokenizer.from_pretrained(self.current_model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.current_model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_response_dialogpt(self, user_input, chat_history=""):
        """Generate response menggunakan DialoGPT"""
        try:
            # Encode input dengan history
            input_text = f"{chat_history}{user_input}"
            input_ids = self.tokenizer.encode(input_text + self.tokenizer.eos_token, return_tensors='pt')
            
            # Generate response
            with torch.no_grad():
                output = self.model.generate(
                    input_ids, 
                    max_length=input_ids.shape[1] + 100,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    attention_mask=torch.ones(input_ids.shape)
                )
            
            # Decode response
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            response = response[len(input_text):].strip()
            
            return response if response else "Maaf, saya tidak mengerti. Bisakah Anda menjelaskan lebih detail?"
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Maaf, terjadi error saat memproses pesan Anda."
    
    def generate_response_pipeline(self, user_input):
        """Generate response menggunakan pipeline"""
        try:
            if "blenderbot" in self.current_model_name:
                # Untuk BlenderBot
                response = self.generator([user_input])
                return response[0]['generated_text'][-1]['content']
                
            else:
                # Untuk model text-generation
                prompt = f"Human: {user_input}\nAssistant:"
                response = self.generator(
                    prompt, 
                    max_length=len(prompt.split()) + 100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=50256
                )
                
                generated_text = response[0]['generated_text']
                assistant_response = generated_text.split("Assistant:")[-1].strip()
                return assistant_response
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Maaf, terjadi error saat memproses pesan Anda."
    
    async def generate_ai_response(self, user_input):
        """Generate AI response secara asynchronous"""
        loop = asyncio.get_event_loop()
        
        if self.generator:
            return await loop.run_in_executor(
                self.executor, 
                self.generate_response_pipeline, 
                user_input
            )
        else:
            return await loop.run_in_executor(
                self.executor, 
                self.generate_response_dialogpt, 
                user_input
            )

# Initialize bot instance
bot_instance = None

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler untuk command /start"""
    welcome_message = """
🤖 Halo! Saya adalah AI Bot gratis yang menggunakan model dari Hugging Face!

📋 Commands yang tersedia:
/start - Tampilkan pesan ini
/help - Bantuan
/models - Lihat model yang tersedia
/change_model - Ganti model AI
/info - Info tentang model saat ini

💬 Kirim pesan apa saja dan saya akan merespon menggunakan AI!
    """
    await update.message.reply_text(welcome_message)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler untuk command /help"""
    help_message = """
🆘 Bantuan Telegram AI Bot

🎯 Cara menggunakan:
1. Kirim pesan biasa untuk chat dengan AI
2. Gunakan /change_model untuk ganti model AI
3. Gunakan /models untuk lihat model tersedia

⚡ Tips:
- Model ringan: Respon lebih cepat
- Model berat: Hasil lebih baik (butuh GPU)
- Tunggu beberapa detik untuk respon AI

🔧 Troubleshooting:
- Jika bot tidak respon, coba /start
- Jika error, model akan otomatis switch ke backup
    """
    await update.message.reply_text(help_message)

async def models_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler untuk command /models"""
    models_text = "🤖 Model AI yang tersedia:\n\n"
    for model_name, description in bot_instance.model_options.items():
        status = "✅ (Aktif)" if model_name == bot_instance.current_model_name else ""
        models_text += f"• {model_name}\n  {description} {status}\n\n"
    
    models_text += "Gunakan /change_model untuk mengganti model"
    await update.message.reply_text(models_text)

async def change_model_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler untuk command /change_model"""
    if len(context.args) == 0:
        await update.message.reply_text(
            "Gunakan: /change_model <nomor>\n\n" +
            "Lihat daftar model dengan /models"
        )
        return
    
    try:
        model_index = int(context.args[0]) - 1
        model_names = list(bot_instance.model_options.keys())
        
        if 0 <= model_index < len(model_names):
            new_model = model_names[model_index]
            await update.message.reply_text(f"🔄 Mengganti model ke: {new_model}\nMohon tunggu...")
            
            bot_instance.current_model_name = new_model
            bot_instance.load_model()
            
            await update.message.reply_text(f"✅ Model berhasil diganti ke: {new_model}")
        else:
            await update.message.reply_text("❌ Nomor model tidak valid. Lihat /models")
            
    except ValueError:
        await update.message.reply_text("❌ Masukkan nomor yang valid")

async def info_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler untuk command /info"""
    gpu_status = "✅ GPU" if torch.cuda.is_available() else "❌ CPU only"
    
    info_text = f"""
📊 Info Model Saat Ini:
🤖 Model: {bot_instance.current_model_name}
💻 Device: {gpu_status}
📝 Deskripsi: {bot_instance.model_options.get(bot_instance.current_model_name, "Unknown")}

⚙️ System Info:
🐍 PyTorch: {torch.__version__}
🔥 CUDA Available: {torch.cuda.is_available()}
    """
    
    await update.message.reply_text(info_text)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler untuk pesan biasa"""
    user_message = update.message.text
    user_name = update.effective_user.first_name
    
    # Tampilkan typing indicator
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, 
        action="typing"
    )
    
    try:
        # Generate AI response
        ai_response = await bot_instance.generate_ai_response(user_message)
        
        # Kirim respon
        await update.message.reply_text(
            f"🤖 {ai_response}",
            parse_mode='HTML'
        )
        
    except Exception as e:
        logger.error(f"Error handling message: {e}")
        await update.message.reply_text(
            "❌ Maaf, terjadi error. Silakan coba lagi atau ganti model dengan /change_model"
        )

def main():
    """Fungsi utama untuk menjalankan bot"""
    global bot_instance
    
    # Masukkan token bot Telegram Anda di sini
    TELEGRAM_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN_HERE"
    
    if TELEGRAM_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN_HERE":
        print("❌ Silakan masukkan TOKEN Telegram bot Anda!")
        print("🔧 Cara mendapat token:")
        print("1. Chat dengan @BotFather di Telegram")
        print("2. Ketik /newbot")
        print("3. Ikuti instruksi")
        print("4. Copy token dan paste di TELEGRAM_TOKEN")
        return
    
    # Initialize bot
    bot_instance = FreeAIBot(TELEGRAM_TOKEN)
    
    # Setup application
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("models", models_command))
    application.add_handler(CommandHandler("change_model", change_model_command))
    application.add_handler(CommandHandler("info", info_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    print("🚀 Bot dimulai! Tekan Ctrl+C untuk stop.")
    print(f"🤖 Model aktif: {bot_instance.current_model_name}")
    
    # Run bot
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
