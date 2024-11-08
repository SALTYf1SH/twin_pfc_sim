import itasca

def kaiwa_new():
    try:
        # Restore the model
        itasca.command("model restore 'pingheng'")
        
        # Reset ball attributes
        itasca.command("ball attribute velocity 0 spin 0 displacement 0")
        
        # Define step interval
        step_interval = 30000
        
        # Get secNum from FISH (assuming it's defined)
        secNum = itasca.fish.get('secNum')
        
        # Loop through sections
        for i in range(5, secNum-2):
            section_name = str(i)
            name = 'result' + str(i)
            
            # Loop through all balls
            for ball in list(itasca.ball.list()):  # Convert to list to avoid iterator issues
                if ball.valid():  # Check if ball is still valid
                    if ball.in_group('1', 'layer') and ball.in_group(f'{section_name}', 'section'):
                        try:
                            ball.delete()
                        except Exception as e:
                            print(f"Error deleting ball {ball.id()}: {str(e)}")
            
            # Save model and solve
            if i < secNum-3:
                itasca.command(f"model save '{name}'")
                itasca.command(f"model solve cycle {step_interval} or ratio-average 1e-3")
            else:
                itasca.command(f"model save '{name}'")
                itasca.command(f"model solve ratio-average 1e-3")
        
        # Save final model
        itasca.command("model save 'final'")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Run the function
try:
    kaiwa_new()
    print("Simulation completed successfully")
    
except Exception as e:
    print(f"Error in main execution: {str(e)}")
