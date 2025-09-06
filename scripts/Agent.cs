using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class AdvancedPathfindingAgent : Agent
{
    [Header("Movement Settings")]
    [SerializeField] private float moveSpeed = 5f;
    [SerializeField] private float turnSpeed = 200f;
    [SerializeField] private float animationSpeed = 1f;
    
    [Header("Training Settings")]
    [SerializeField] private Transform target;
    [SerializeField] private float targetReachedDistance = 2f;
    [SerializeField] private float maxEpisodeTime = 120f; // Longer episodes
    [SerializeField] private bool continuousTargetTracking = true;
    
    [Header("Detection")]
    [SerializeField] private float rayDistance = 10f;
    [SerializeField] private LayerMask obstacleLayerMask = -1;
    
    [Header("Rewards")]
    [SerializeField] private float reachTargetReward = 5f; // Lower since we don't restart
    [SerializeField] private float hitTargetReward = 8f;
    [SerializeField] private float missedHitPenalty = -1f;
    [SerializeField] private float timeStepPenalty = -0.0005f;
    [SerializeField] private float stayNearTargetReward = 0.01f;
    [SerializeField] private float movementEfficiencyReward = 0.005f;
    
    [Header("Hit Settings")]
    [SerializeField] private float hitRange = 3f;
    [SerializeField] private float hitCooldown = 2f;
    
    [Header("Animation")]
    [SerializeField] private Animator animator;
    [SerializeField] private string moveSpeedParameter = "MoveSpeed";
    [SerializeField] private string isMovingParameter = "IsMoving";
    [SerializeField] private string hitTriggerParameter = "Hit";
    [SerializeField] private string celebrateTriggerParameter = "Celebrate";
    
    [Header("Target Management")]
    [SerializeField] private float targetMoveRange = 5f;
    [SerializeField] private float targetMoveInterval = 10f;
    [SerializeField] private bool randomTargetMovement = false;
    
    // Private variables
    private Rigidbody rb;
    private Vector3 startingPosition;
    private Quaternion startingRotation;
    private Vector3 previousPosition;
    private float episodeTimer;
    private float lastHitTime;
    private float lastTargetMoveTime;
    private float timeNearTarget;
    private int targetsReached;
    private bool isNearTarget;
    private Vector3 lastTargetPosition;
    
    // Animation states
    private bool isCurrentlyMoving;
    private float currentMoveSpeed;
    
    public override void Initialize()
    {
        rb = GetComponent<Rigidbody>();
        
        // Get animator if not assigned
        if (animator == null)
            animator = GetComponent<Animator>();
            
        startingPosition = transform.position;
        startingRotation = transform.rotation;
        
        // Set up rigidbody for flat movement
        if (rb != null)
        {
            rb.constraints = RigidbodyConstraints.FreezePositionY | 
                            RigidbodyConstraints.FreezeRotationX | 
                            RigidbodyConstraints.FreezeRotationZ;
        }
    }
    
    public override void OnEpisodeBegin()
    {
        // Reset agent
        transform.position = startingPosition;
        transform.rotation = startingRotation;
        
        if (rb != null)
        {
            rb.velocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
        }
        
        // Reset episode variables
        episodeTimer = 0f;
        lastHitTime = -hitCooldown;
        lastTargetMoveTime = 0f;
        timeNearTarget = 0f;
        targetsReached = 0;
        isNearTarget = false;
        isCurrentlyMoving = false;
        currentMoveSpeed = 0f;
        previousPosition = transform.position;
        
        // Set initial target position
        // SetRandomTarget();
        lastTargetPosition = target != null ? target.position : Vector3.zero;
        
        // Reset animations
        UpdateAnimations();
    }
    
    private void SetRandomTarget()
    {
        if (target != null)
        {
            Vector3 randomPosition = startingPosition + new Vector3(
                Random.Range(-targetMoveRange, targetMoveRange),
                0f,
                Random.Range(-targetMoveRange, targetMoveRange)
            );
            target.position = randomPosition;
            lastTargetPosition = randomPosition;
            lastTargetMoveTime = episodeTimer;
        }
    }
    
    private void MoveTargetRandomly()
    {
        if (target != null && randomTargetMovement)
        {
            // Move target to a new random position
            Vector3 currentTargetPos = target.position;
            Vector3 randomOffset = new Vector3(
                Random.Range(-5f, 5f),
                0f,
                Random.Range(-5f, 5f)
            );
            
            Vector3 newPosition = currentTargetPos + randomOffset;
            
            // Keep within bounds
            newPosition.x = Mathf.Clamp(newPosition.x, 
                startingPosition.x - targetMoveRange, 
                startingPosition.x + targetMoveRange);
            newPosition.z = Mathf.Clamp(newPosition.z, 
                startingPosition.z - targetMoveRange, 
                startingPosition.z + targetMoveRange);
                
            target.position = newPosition;
            lastTargetPosition = newPosition;
            lastTargetMoveTime = episodeTimer;
            
            Debug.Log("Target moved to new position!");
        }
    }
    
    public override void CollectObservations(VectorSensor vectorSensor)
    {
        int observationCount = 0;
        
        // Agent's local position to target (2 values - X, Z only)
        if (target != null)
        {
            Vector3 localPos = transform.InverseTransformPoint(target.position);
            vectorSensor.AddObservation(localPos.x);
            vectorSensor.AddObservation(localPos.z);
        }
        else
        {
            vectorSensor.AddObservation(0f);
            vectorSensor.AddObservation(0f);
        }
        observationCount += 2;
        
        // Distance to target (1 value)
        float distance = target != null ? Vector3.Distance(transform.position, target.position) : 0f;
        vectorSensor.AddObservation(distance);
        observationCount += 1;
        
        // Agent's velocity in local space (2 values)
        if (rb != null)
        {
            Vector3 localVelocity = transform.InverseTransformDirection(rb.velocity);
            vectorSensor.AddObservation(localVelocity.x);
            vectorSensor.AddObservation(localVelocity.z);
        }
        else
        {
            vectorSensor.AddObservation(0f);
            vectorSensor.AddObservation(0f);
        }
        observationCount += 2;
        
        // Raycast in 8 directions for better obstacle detection
        Vector3[] directions = {
            transform.forward,                           // Forward
            -transform.forward,                          // Back
            -transform.right,                            // Left
            transform.right,                             // Right
            (transform.forward + transform.right).normalized,    // Forward-Right
            (transform.forward - transform.right).normalized,    // Forward-Left
            (-transform.forward + transform.right).normalized,   // Back-Right
            (-transform.forward - transform.right).normalized    // Back-Left
        };
        
        for (int i = 0; i < 8; i++)
        {
            float rayDistance = this.rayDistance;
            if (Physics.Raycast(transform.position, directions[i], out RaycastHit hit, rayDistance, obstacleLayerMask))
            {
                vectorSensor.AddObservation(hit.distance / rayDistance);
            }
            else
            {
                vectorSensor.AddObservation(1f);
            }
            observationCount += 1;
        }
        
        // Episode progress (1 value)
        vectorSensor.AddObservation(episodeTimer / maxEpisodeTime);
        observationCount += 1;
        
        // Hit cooldown status (1 value)
        float timeSinceLastHit = episodeTimer - lastHitTime;
        float hitCooldownProgress = Mathf.Clamp01(timeSinceLastHit / hitCooldown);
        vectorSensor.AddObservation(hitCooldownProgress);
        observationCount += 1;
        
        // Can hit target (1 value)
        bool canHitTarget = target != null && distance <= hitRange && timeSinceLastHit >= hitCooldown;
        vectorSensor.AddObservation(canHitTarget ? 1f : 0f);
        observationCount += 1;
        
        // Targets reached this episode (1 value - normalized)
        vectorSensor.AddObservation(targetsReached / 10f); // Normalize by expected max targets
        observationCount += 1;
        
        // Time near target (1 value)
        vectorSensor.AddObservation(timeNearTarget / maxEpisodeTime);
        observationCount += 1;
        
        // Target movement velocity (2 values)
        if (target != null)
        {
            Vector3 targetVelocity = (target.position - lastTargetPosition) / Time.fixedDeltaTime;
            Vector3 localTargetVel = transform.InverseTransformDirection(targetVelocity);
            vectorSensor.AddObservation(localTargetVel.x);
            vectorSensor.AddObservation(localTargetVel.z);
        }
        else
        {
            vectorSensor.AddObservation(0f);
            vectorSensor.AddObservation(0f);
        }
        observationCount += 2;
        
        // Debug: Print observation count (only once per episode)
        if (episodeTimer < 0.1f)
        {
            Debug.Log($"Advanced Agent - Total observations: {observationCount}");
        }
    }
    
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        // Get movement actions
        float moveForward = actionBuffers.ContinuousActions[0];  // Forward/Back (-1 to 1)
        float moveRight = actionBuffers.ContinuousActions[1];    // Left/Right (-1 to 1)
        float turn = actionBuffers.ContinuousActions[2];         // Turn Left/Right (-1 to 1)
        
        // Get hit action (discrete action) - with safety check
        bool hitAction = false;
        if (actionBuffers.DiscreteActions.Length > 0)
        {
            hitAction = actionBuffers.DiscreteActions[0] == 1;
        }
        
        // Apply movement
        MoveAgent(moveForward, moveRight, turn);
        
        // Handle hit action
        if (hitAction)
        {
            HandleHitAction();
        }
        
        // Update animations
        UpdateAnimations();
        
        // Calculate rewards
        CalculateRewards();
        
        // Check for target movement
        CheckTargetMovement();
        
        // Check if episode should end (only for time limit or falling)
        CheckEpisodeEnd();
        
        episodeTimer += Time.fixedDeltaTime;
        
        // Update target position tracking
        if (target != null)
        {
            lastTargetPosition = target.position;
        }
    }
    
    private void MoveAgent(float forward, float right, float turn)
    {
        if (rb == null) return;
        
        // Calculate movement in local space
        Vector3 movement = Vector3.zero;
        movement += transform.forward * forward * moveSpeed;  // Forward/Back
        movement += transform.right * right * moveSpeed;      // Left/Right
        
        // Apply movement
        Vector3 newPosition = transform.position + movement * Time.fixedDeltaTime;
        newPosition.y = transform.position.y; // Keep Y locked
        rb.MovePosition(newPosition);
        
        // Apply turning
        if (Mathf.Abs(turn) > 0.01f)
        {
            float rotation = turn * turnSpeed * Time.fixedDeltaTime;
            Quaternion deltaRotation = Quaternion.Euler(0f, rotation, 0f);
            rb.MoveRotation(rb.rotation * deltaRotation);
        }
        
        // Update movement state for animations
        float totalMovement = Mathf.Abs(forward) + Mathf.Abs(right);
        currentMoveSpeed = totalMovement;
        isCurrentlyMoving = totalMovement > 0.1f;
    }
    
    private void UpdateAnimations()
    {
        if (animator == null) return;
        
        // Update movement animations
        if (!string.IsNullOrEmpty(moveSpeedParameter))
        {
            animator.SetFloat(moveSpeedParameter, currentMoveSpeed * animationSpeed);
        }
        
        if (!string.IsNullOrEmpty(isMovingParameter))
        {
            animator.SetBool(isMovingParameter, isCurrentlyMoving);
        }
    }
    
    private void HandleHitAction()
    {
        // Check if hit is on cooldown
        float timeSinceLastHit = episodeTimer - lastHitTime;
        if (timeSinceLastHit < hitCooldown)
        {
            // Hit is on cooldown, small penalty
            AddReward(-0.1f);
            return;
        }
        
        // Update hit time
        lastHitTime = episodeTimer;
        
        // Trigger hit animation
        if (animator != null && !string.IsNullOrEmpty(hitTriggerParameter))
        {
            animator.SetTrigger(hitTriggerParameter);
        }
        
        if (target == null) return;
        
        // Check if target is within hit range
        float distanceToTarget = Vector3.Distance(transform.position, target.position);
        
        if (distanceToTarget <= hitRange)
        {
            // Successful hit!
            AddReward(hitTargetReward);
            
            // Visual feedback
            print("HIT! Target successfully hit!");
            Debug.Log("Target Hit!");
            
            // Don't end episode, just move target to new position
            SetRandomTarget();
            targetsReached++;
            
            // Trigger celebration animation
            if (animator != null && !string.IsNullOrEmpty(celebrateTriggerParameter))
            {
                animator.SetTrigger(celebrateTriggerParameter);
            }
        }
        else
        {
            // Missed hit
            AddReward(missedHitPenalty);
        }
    }
    
    private void CalculateRewards()
    {
        if (target == null) return;
        
        // Small time penalty to encourage efficiency
        AddReward(timeStepPenalty);
        
        float currentDistance = Vector3.Distance(transform.position, target.position);
        float previousDistance = Vector3.Distance(previousPosition, target.position);
        
        // Reward for getting closer to target
        if (currentDistance < previousDistance)
        {
            float improvement = previousDistance - currentDistance;
            AddReward(improvement * movementEfficiencyReward);
        }
        
        // Reward for staying near target
        if (currentDistance <= targetReachedDistance)
        {
            if (!isNearTarget)
            {
                // First time reaching target
                AddReward(reachTargetReward);
                isNearTarget = true;
                print("Target reached! Continuing to track...");
            }
            
            // Continuous reward for staying near target
            timeNearTarget += Time.fixedDeltaTime;
            AddReward(stayNearTargetReward);
        }
        else
        {
            isNearTarget = false;
        }
        
        // Store position for next frame
        previousPosition = transform.position;
        
        // Small penalty for being far from target
        AddReward(-currentDistance * 0.0005f);
    }
    
    private void CheckTargetMovement()
    {
        if (randomTargetMovement && target != null)
        {
            // Move target periodically
            if (episodeTimer - lastTargetMoveTime >= targetMoveInterval)
            {
                MoveTargetRandomly();
            }
        }
    }
    
    private void CheckEpisodeEnd()
    {
        // Only end episode for time limit or falling off platform
        
        // Time limit reached
        if (episodeTimer >= maxEpisodeTime)
        {
            Debug.Log($"Episode ended. Targets reached: {targetsReached}");
            EndEpisode();
            return;
        }
        
        // Fell off platform
        if (transform.position.y < startingPosition.y - 2f)
        {
            AddReward(-10f);
            EndEpisode();
        }
    }
    
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        var discreteActionsOut = actionsOut.DiscreteActions;
        
        if (continuousActionsOut.Length >= 3)
        {
            // WASD movement
            continuousActionsOut[0] = Input.GetAxis("Vertical");     // W/S - Forward/Back
            continuousActionsOut[1] = Input.GetAxis("Horizontal");   // A/D - Left/Right
            
            // Q/E for turning
            float turn = 0f;
            if (Input.GetKey(KeyCode.Q)) turn = -1f;  // Turn left
            if (Input.GetKey(KeyCode.E)) turn = 1f;   // Turn right
            continuousActionsOut[2] = turn;
        }
        
        // Spacebar for hitting
        if (discreteActionsOut.Length >= 1)
        {
            discreteActionsOut[0] = Input.GetKeyDown(KeyCode.Space) ? 1 : 0;
        }
    }
    
    // Collision detection for additional interactions
    private void OnTriggerEnter(Collider other)
    {
        if (other.transform == target && !isNearTarget)
        {
            isNearTarget = true;
            AddReward(reachTargetReward);
            print("Target reached via collision!");
        }
    }
    
    // Visualize in Scene view
    private void OnDrawGizmosSelected()
    {
        if (target != null)
        {
            // Line to target
            Gizmos.color = Color.green;
            Gizmos.DrawLine(transform.position, target.position);
            
            // Target area
            Gizmos.color = Color.red;
            Gizmos.DrawWireSphere(target.position, targetReachedDistance);
            
            // Target movement range
            Gizmos.color = Color.blue;
            Gizmos.DrawWireCube(startingPosition, new Vector3(targetMoveRange * 2, 0.1f, targetMoveRange * 2));
        }
        
        // Hit range visualization
        Gizmos.color = Color.yellow;
        Gizmos.DrawWireSphere(transform.position, hitRange);
        
        // Raycast directions (8 directions)
        if (Application.isPlaying)
        {
            Gizmos.color = Color.cyan;
            Vector3[] directions = {
                transform.forward,                           // Forward
                -transform.forward,                          // Back
                -transform.right,                            // Left
                transform.right,                             // Right
                (transform.forward + transform.right).normalized,    // Forward-Right
                (transform.forward - transform.right).normalized,    // Forward-Left
                (-transform.forward + transform.right).normalized,   // Back-Right
                (-transform.forward - transform.right).normalized    // Back-Left
            };
            
            foreach (var direction in directions)
            {
                Gizmos.DrawRay(transform.position, direction * rayDistance * 0.5f);
            }
        }
    }
}

