# myapp/views.py
from django.shortcuts import render, redirect
from .models import PressedButton
from .forms import PressButtonForm
from django.utils import timezone
from .ImageReconstruction import getData
def home(request):
    if request.method == 'POST':
        # Handle button press
        result_number = getData()  # Your custom function
        PressedButton.objects.create(result_number=result_number)
        return redirect('home')

    form = PressButtonForm()
    button_presses = PressedButton.objects.all()
    return render(request, 'myDashboard/home.html', {'form': form, 'button_presses': button_presses})
