resource "aws_ecs_cluster" "api" {
  name = "${local.name_prefix}-api"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = local.tags
}

resource "aws_ecs_task_definition" "api" {
  family                   = "${local.name_prefix}-api"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = 512
  memory                   = 1024
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn
  tags                     = local.tags

  container_definitions = jsonencode([
    {
      name      = "api"
      image     = var.api_image
      essential = true
      portMappings = [{
        containerPort = var.container_port
        protocol      = "tcp"
      }]
      environment = [
        { name = "AWS_REGION", value = var.region },
        { name = "CA_REGION", value = var.region },
        { name = "CA_SAGEMAKER_ENDPOINT", value = var.sagemaker_endpoint },
        { name = "CA_GUARDRAIL_ID", value = var.guardrail_id },
        { name = "CA_CORS_ORIGINS", value = "http://localhost:3000" },
        { name = "PYTHONPATH", value = "/app" },
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = "/aws/ecs/${local.name_prefix}-api"
          "awslogs-region"        = var.region
          "awslogs-stream-prefix" = "api"
        }
      }
      healthCheck = {
        command     = ["CMD-SHELL", "python -c \"import urllib.request;urllib.request.urlopen('http://127.0.0.1:${var.container_port}/health').read()\""]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 20
      }
    }
  ])
}

# --- Load balancer -------------------------------------------------------

resource "aws_security_group" "alb" {
  name        = "${local.name_prefix}-api-alb-sg"
  description = "Allow public HTTP to the API load balancer"
  vpc_id      = var.vpc_id
  tags        = local.tags

  ingress {
    description = "HTTP from the internet"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "task" {
  name        = "${local.name_prefix}-api-task-sg"
  description = "Allow the load balancer to reach the API tasks"
  vpc_id      = var.vpc_id
  tags        = local.tags

  ingress {
    description     = "HTTP from the load balancer"
    from_port       = var.container_port
    to_port         = var.container_port
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_lb" "api" {
  name               = substr("${local.name_prefix}-api", 0, 32)
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = var.public_subnet_ids
  tags               = local.tags
}

resource "aws_lb_target_group" "api" {
  name        = substr("${local.name_prefix}-api", 0, 32)
  port        = var.container_port
  protocol    = "HTTP"
  target_type = "ip"
  vpc_id      = var.vpc_id
  tags        = local.tags

  health_check {
    path     = "/health"
    matcher  = "200"
    interval = 30
    timeout  = 5
  }
}

resource "aws_lb_listener" "api" {
  load_balancer_arn = aws_lb.api.arn
  port              = 80
  protocol          = "HTTP"
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
}

resource "aws_ecs_service" "api" {
  name            = "${local.name_prefix}-api"
  cluster         = aws_ecs_cluster.api.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [aws_security_group.task.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "api"
    container_port   = var.container_port
  }

  depends_on = [aws_lb_listener.api]
  tags       = local.tags
}

output "api_url" {
  value = "http://${aws_lb.api.dns_name}"
}
